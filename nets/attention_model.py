import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from utils.tensor_functions import compute_in_batches
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many, adapt_top


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 num_agents=2,
                 use_rope=False,
                 rope_theta=1e4,
                 use_entropy_scheduling=False,
                 **kwargs):
        super(AttentionModel, self).__init__()

        # Problem parameters
        self.num_agents = num_agents
        self.problem = problem
        
        # Entropy-guided scheduling for multi-agent decoding
        self.use_entropy_scheduling = use_entropy_scheduling

        # Dimensions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder parameters
        self.n_heads = n_heads
        self.n_encode_layers = n_encode_layers
        self.checkpoint_encoder = checkpoint_encoder

        # Decoder parameters
        self.temp = 1.0
        self.decode_type = None
        self.shrink_size = shrink_size
        self.tanh_clipping = tanh_clipping

        # Mask parameters
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
        # 剩余容量 / 剩余长度 / 剩余奖励
        step_context_dim = embedding_dim + 1

        # Node dimension
        node_dim = 3  # x, y, prize

        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        # Encoder embeddings
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        # 图注意力编码器
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            use_rope=use_rope,
            rope_theta=rope_theta
        )

        # Decoder embeddings
        assert embedding_dim % n_heads == 0
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        # 投影层
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        # Encoder
        # 编码节点嵌入
        pos_xy = self._build_pos_xy(input)
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            def _encode(e, p):
                return self.embedder(e, pos_xy=p)[0]
            embeddings = checkpoint(_encode, self._init_embed(input), pos_xy)
        else:
            embeddings, _ = self.embedder(self._init_embed(input), pos_xy=pos_xy)

        # Decode log_probs and paths
        # 解码路径
        _log_p, pi = self._inner(input, embeddings)

        # Get costs
        # 计算成本
        cost, mask = self.problem.get_costs(input, pi)

        # Log likelihood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        #计算对数似然
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        # Adapt for TOP
        ll, pi, cost = adapt_top(ll, pi, cost=cost)

        # Output costs (cost), log likelihoods (ll), and maybe routes (pi)
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        pos_xy = self._build_pos_xy(input)
        embeddings, _ = self.embedder(self._init_embed(input), pos_xy=pos_xy)
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None
        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        """Caculate log likelihood for loss function."""

        # Get log_p corresponding to selected actions of each agent
        log_p = tuple()
        for k in range(self.num_agents):
            log_p = log_p + (_log_p[:, :, k].gather(2, a[:, :, k].unsqueeze(-1)), )

            # Optional: mask out actions irrelevant to objective so they do not get reinforced
            if mask is not None:
                log_p[mask] = 0

            assert (log_p[k] > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        log_p = torch.cat(log_p, dim=2)
        return log_p.sum(1)

    def _init_embed(self, input):
        """Embedding for the inputs"""
        features = ('prize',)
        end_depot = 'depot2' if 'depot2' in input else 'depot'
        embeddings = (
            self.init_embed_depot(input['depot'])[:, None, :],
            self.init_embed(torch.cat((
                input['loc'],
                *(input[feat][:, :, None] for feat in features)
            ), -1)),
            self.init_embed_depot(input[end_depot])[:, None, :]
        )
        return torch.cat(embeddings, 1)

    def _inner(self, input, embeddings):
        """Make predictions."""

        # Initialize output probabilities and chosen indexes
        outputs, sequences = [], []

        # Initialize problem state
        state = self.problem.make_state(input, num_agents=self.num_agents)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Batch dimension
        batch_size, graph_size, _ = state.coords.size()

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well, and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            # Predict probabilities for each node
            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences
            if self.use_entropy_scheduling:
                # Entropy-guided dynamic scheduling (slower but better quality)
                agent_order = self._compute_entropy_priority(log_p, mask)  # [batch_size, num_agents]
                selected, log_p_aux = self._select_nodes_dynamic_order(log_p, mask, agent_order, batch_size)
            else:
                # Fixed order (original fast implementation)
                selected, log_p_aux = tuple(), log_p.clone()
                for k in range(self.num_agents):
                    agent_selection = self._select_node(log_p_aux.exp()[:, 0, k, :], mask[k][:, 0, :])
                    if k < self.num_agents - 1:
                        aux = log_p_aux[..., -1].clone()
                        log_p_aux[:, 0, k + 1:][[b for b in range(batch_size)], :, agent_selection] = -math.inf
                        log_p_aux[..., -1] = aux  # End depot can always be visited
                        log_p_aux[:, 0, k + 1:] = torch.log_softmax(log_p_aux[:, 0, k + 1:] / self.temp, dim=-1)
                    selected = selected + (agent_selection[:, None], )
                selected = torch.cat(selected, dim=1)
            
            log_p = log_p_aux

            # Update state
            state = state.update(selected)

            # Now make log_p and selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        pos_xy = self._build_pos_xy(input)
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input), pos_xy=pos_xy)[0]),  # Pack input with embeddings
            batch_rep, iter_rep, self.num_agents
        )

    def _build_pos_xy(self, input):
        end_depot = 'depot2' if 'depot2' in input else 'depot'
        return torch.cat((
            input['depot'][:, None, :],
            input['loc'],
            input[end_depot][:, None, :]
        ), dim=1)

    def _select_node(self, probs, mask):
        """ArgMax or sample from probabilities to select next node."""
        assert (probs == probs).all(), "Probs should not contain any nans"

        # ArgMax
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        # Sample
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _compute_entropy_priority(self, log_p, mask):
        """
        Compute decision entropy for each agent and return dynamic priority order.
        Lower entropy = more certain decision = higher priority (earlier selection).
        
        Args:
            log_p: [batch_size, 1, num_agents, graph_size] - log probabilities for each agent
            mask: list of [batch_size, 1, graph_size] for each agent - True means invalid
        
        Returns:
            agent_order: [batch_size, num_agents] - agent indices sorted by entropy (ascending)
        """
        batch_size = log_p.size(0)
        num_agents = log_p.size(2)
        
        # Compute entropy for each agent in parallel
        entropies = []
        for k in range(num_agents):
            # Get probability distribution for agent k
            log_p_k = log_p[:, 0, k, :]  # [batch_size, graph_size]
            p_k = log_p_k.exp()  # Convert to probabilities
            
            # Apply mask: only consider valid nodes
            # mask[k][:, 0, :] is [batch_size, graph_size] where True = invalid
            valid_mask = ~mask[k][:, 0, :]  # Invert: True = valid
            p_k_valid = p_k * valid_mask.float()
            
            # Renormalize (since masking changes probability sum)
            p_k_sum = p_k_valid.sum(dim=-1, keepdim=True)
            p_k_valid = p_k_valid / (p_k_sum + 1e-10)  # Avoid division by zero
            
            # Compute entropy: H = -Σ(p * log(p))
            # Add small epsilon to avoid log(0)
            log_p_k_valid = torch.log(p_k_valid + 1e-10)
            entropy_k = -(p_k_valid * log_p_k_valid).sum(dim=-1)  # [batch_size]
            
            entropies.append(entropy_k)
        
        # Stack: [batch_size, num_agents]
        entropies = torch.stack(entropies, dim=1)
        
        # Sort agents by entropy (ascending order: low entropy first)
        agent_order = torch.argsort(entropies, dim=1)  # [batch_size, num_agents]
        
        return agent_order

    def _select_nodes_dynamic_order(self, log_p, mask, agent_order, batch_size):
        """
        Select nodes for each agent according to dynamic order.
        Note: This is slower than fixed order due to per-batch processing.
        
        Args:
            log_p: [batch_size, 1, num_agents, graph_size]
            mask: list of [batch_size, 1, graph_size] for each agent
            agent_order: [batch_size, num_agents] - dynamic agent order for each batch
            batch_size: int
        
        Returns:
            selected: [batch_size, num_agents] - selected nodes for each agent
        """
        # Prepare tensors in agent-order space so we can reuse vectorised selection logic
        mask_tensor = torch.stack(mask, dim=2).to(torch.bool)  # [batch, 1, num_agents, graph]
        expanded_indices = agent_order[:, None, :, None].expand(-1, 1, self.num_agents, log_p.size(-1))

        log_p_perm = torch.gather(log_p.clone(), 2, expanded_indices)
        mask_perm = torch.gather(mask_tensor, 2, expanded_indices)

        selected_perm = []
        log_p_aux = log_p_perm.clone()

        for k in range(self.num_agents):
            probs = log_p_aux.exp()[:, 0, k, :]
            mask_k = mask_perm[:, 0, k, :]
            agent_selection = self._select_node(probs, mask_k)

            selected_perm.append(agent_selection[:, None])

            if k < self.num_agents - 1:
                aux = log_p_aux[..., -1].clone()
                batch_indices = torch.arange(batch_size, device=log_p.device)
                log_p_aux[:, 0, k + 1:][batch_indices, :, agent_selection] = -math.inf
                log_p_aux[..., -1] = aux
                log_p_aux[:, 0, k + 1:] = torch.log_softmax(log_p_aux[:, 0, k + 1:] / self.temp, dim=-1)

        selected_perm = torch.cat(selected_perm, dim=1)

        selected = torch.zeros_like(selected_perm)
        selected.scatter_(1, agent_order, selected_perm)

        inverse_order = torch.argsort(agent_order, dim=1)
        scatter_idx = inverse_order[:, None, :, None].expand(-1, 1, self.num_agents, log_p.size(-1))
        log_p_orig_order = torch.gather(log_p_aux, 2, scatter_idx)

        return selected, log_p_orig_order

    def _precompute(self, embeddings, num_steps=1):
        """Precompute encoder embeddings."""

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        """Top K probabilities."""
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        """Predict probabilities."""

        # Compute query = context node embedding
        query = tuple()
        for k in range(self.num_agents):
            query = query + (
                (fixed.context_node_projected +
                 self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, k))
                 )[:, :, None, :],
            )
        query = torch.cat(query, dim=2)

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Get the mask
        mask = [state.get_mask(k) for k in range(self.num_agents)]

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, agent_id, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        # Get info from current node
        current_node = state.get_current_node(agent_id)
        batch_size, num_steps = current_node.size()
        return torch.cat(
            (
                torch.gather(
                    embeddings,
                    1,
                    current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1)),
                (
                    (state.get_remaining_length(agent_id).view(-1, batch_size) / state.max_length[..., 0])
                        .view(batch_size, -1)[:, :, None]
                )
            ),
            -1
        )

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """Multi-Head Attention mechanism + Single-Head Attention mechanism."""

        # Dimensions
        batch_size, num_steps, _, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Rearrange dimensions: (n_heads, batch_size, num_steps, num_agents, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, self.num_agents, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            for k in range(self.num_agents):
                compatibility[..., k, :][mask[k][None, :, :, :].expand_as(compatibility[..., k, :])] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, num_agents, embedding_dim)
        final_Q = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, self.num_agents, self.n_heads * val_size))

        # Batch matrix multiplication to get logits (== 'compatibility') (batch_size, num_steps, num_agents, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            for k in range(self.num_agents):
                logits[..., k, :][mask[k]] = -math.inf
        return logits, final_Q

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

import re
import torch
import numpy as np
import os
import json
import torch.nn.functional as F


def load_problem(name):
    from problems import TOP
    problem = {
        'top': TOP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def adapt_top(ll, pi, cost=None):
    if ll.ndim > 1:
        pi = pi.transpose(2, 1).reshape((-1, pi.shape[1] * pi.shape[2]))
        ll = ll.sum(axis=-1)
        # ll = ll.mean(axis=-1)
        if cost is not None:
            cost = cost.sum(axis=-1)
            return ll, pi, cost
        return ll, pi


def torch_load_cpu(load_path):
    """Load a torch checkpoint on CPU with PyTorch 2.6 compatibility.

    PyTorch 2.6 changed torch.load default `weights_only` to True which breaks
    older checkpoints that pickle model classes. We explicitly set
    `weights_only=False` for trusted local checkpoints, with a safe fallback
    for older torch versions that don't support the argument.
    """
    try:
        # Prefer absolute path resolution similar to _load_model_file
        resolved_path = os.path.join(os.getcwd(), load_path) if not os.path.isabs(load_path) else load_path
        return torch.load(resolved_path, map_location=lambda storage, loc: storage, weights_only=False)
    except TypeError:
        # Older torch without weights_only arg
        return torch.load(load_path, map_location=lambda storage, loc: storage)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def _remap_old_to_new_key(k: str) -> str:
    """将旧版权重中的键名映射为当前代码结构的键名。"""
    # MHA block: embedder.layers.{i}.0.module.* -> embedder.layers.{i}.mha.*
    k = re.sub(r'^embedder\.layers\.(\d+)\.0\.module\.', r'embedder.layers.\1.mha.', k)
    # Norm1: embedder.layers.{i}.1.normalizer.* -> embedder.layers.{i}.norm1.normalizer.*
    k = re.sub(r'^embedder\.layers\.(\d+)\.1\.normalizer\.', r'embedder.layers.\1.norm1.normalizer.', k)
    # FF: embedder.layers.{i}.2.module.{0,2}.* -> embedder.layers.{i}.ff.{0,2}.*
    def _ff_sub(m):
        layer_idx = m.group(1)
        sub_idx = m.group(2)
        return f'embedder.layers.{layer_idx}.ff.{sub_idx}.'
    k = re.sub(r'^embedder\.layers\.(\d+)\.2\.module\.(\d)\.', _ff_sub, k)
    # Norm2: embedder.layers.{i}.3.normalizer.* -> embedder.layers.{i}.norm2.normalizer.*
    k = re.sub(r'^embedder\.layers\.(\d+)\.3\.normalizer\.', r'embedder.layers.\1.norm2.normalizer.', k)
    return k


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch_load_cpu(load_path)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    # Build a compatible state dict: map old keys to new keys and filter by existence and shape
    current_state = model.state_dict()
    compatible = {}
    for k, v in load_model_state_dict.items():
        kk = _remap_old_to_new_key(k)
        if kk in current_state and current_state[kk].shape == v.shape:
            compatible[kk] = v

    # Update current state and load
    current_state.update(compatible)
    model.load_state_dict(current_state, strict=False)

    return model, load_optimizer_state_dict




def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None, num_agents=2, use_rope=None, use_entropy_scheduling=None):
    from nets.attention_model import AttentionModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_name = args.get('model', 'attention')
    assert model_name == 'attention', "Only attention model checkpoints are supported"

    resolved_use_rope = args.get('use_rope', False) if use_rope is None else use_rope
    resolved_rope_theta = args.get('rope_theta', 1e4)
    resolved_use_entropy_scheduling = args.get('use_entropy_scheduling', False) if use_entropy_scheduling is None else use_entropy_scheduling

    model = AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),
        num_agents=num_agents,
        use_rope=resolved_use_rope,
        rope_theta=resolved_rope_theta,
        use_entropy_scheduling=resolved_use_entropy_scheduling
    )
    # Overwrite model parameters by parameters to load with in-memory remapping
    load_data = torch_load_cpu(model_filename)
    if isinstance(load_data, dict):
        loaded = load_data.get('model', load_data)
    else:
        loaded = load_data.state_dict()

    current_state = model.state_dict()
    compatible = {}
    for k, v in (loaded.items() if isinstance(loaded, dict) else loaded):
        kk = _remap_old_to_new_key(k)
        if kk in current_state and current_state[kk].shape == v.shape:
            compatible[kk] = v

    current_state.update(compatible)
    model.load_state_dict(current_state, strict=False)

    # Optionally load optimizer etc (also remapped inside)
    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1, num_agents=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi = pi.reshape(pi.shape[0], -1, num_agents)
        cost, mask = get_cost_func(input, pi)
        cost = cost.sum(axis=-1)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, *pi.shape).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

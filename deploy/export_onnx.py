#!/usr/bin/env python3
"""
deploy/export_onnx.py — Export MARGA model to ONNX format for on-device inference.

Two sub-networks are exported:
  encoder.onnx       — Encoder + precompute_fixed (runs once per problem instance)
  decoder_step.onnx  — Single autoregressive decoding step (runs ~30-80 times)

The on-device control flow (greedy selection, entropy scheduling, state update)
is handled by the Android runtime in MARGASolver.kt.

Usage:
  pip install -r deploy/requirements_deploy.txt

  python deploy/export_onnx.py \\
      --ckpt pretrained/Mixed-training/top_mixed100/MARGA/epoch-99.pt \\
      --n_nodes 100 --num_agents 5 \\
      --output deploy/exported/
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nets.attention_model import AttentionModel
from utils import load_problem


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper 1: Encoder
# Outputs node embeddings, precomputed KV caches, coordinates, and per-node
# budget limits — everything the decoder needs that is constant across steps.
# ──────────────────────────────────────────────────────────────────────────────
class EncoderWrapper(nn.Module):
    def __init__(self, model: AttentionModel):
        super().__init__()
        self.model = model

    def forward(self, depot, loc, prize, max_length):
        inp = {'depot': depot, 'loc': loc, 'prize': prize, 'max_length': max_length}

        pos_xy        = self.model._build_pos_xy(inp)
        init_emb      = self.model._init_embed(inp)
        embeddings, _ = self.model.embedder(init_emb, pos_xy=pos_xy)
        fixed         = self.model._precompute(embeddings)

        # coords [B, N+2, 2]: [start_depot, loc_1..N, end_depot]
        coords = torch.cat((depot[:, None, :], loc, depot[:, None, :]), dim=1)

        # node_max_length [B, N+2]: maximum cumulative route length allowed when
        # visiting each node (matches StateTOP.initialize exactly)
        node_max_length = (
            max_length[:, None]
            - (depot[:, None, :] - coords).norm(p=2, dim=-1)
            - 1e-6
        )

        return (
            fixed.node_embeddings,         # [B, N+2, D]
            fixed.context_node_projected,  # [B, 1, D]
            fixed.glimpse_key,             # [H, B, 1, N+2, D/H]
            fixed.glimpse_val,             # [H, B, 1, N+2, D/H]
            fixed.logit_key,               # [B, 1, N+2, D]
            node_max_length,               # [B, N+2]
            coords,                        # [B, N+2, 2]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper 2: Decoder Step
# Computes log-probabilities for all agents in one forward pass.
# Full feasibility mask is computed here to match StateTOP.get_mask exactly:
#   mask = visited | (lengths[k] + dist(cur, node) > node_max_length[node])
# ──────────────────────────────────────────────────────────────────────────────
class DecoderStepWrapper(nn.Module):
    def __init__(self, model: AttentionModel):
        super().__init__()
        self.model     = model
        self.K         = model.num_agents
        self.embed_dim = model.embedding_dim

    def forward(
        self,
        node_embeddings,   # [B, N+2, D]
        fixed_context,     # [B, 1, D]
        glimpse_key,       # [H, B, 1, N+2, D/H]
        glimpse_val,       # [H, B, 1, N+2, D/H]
        logit_key,         # [B, 1, N+2, D]
        node_max_length,   # [B, N+2]
        coords,            # [B, N+2, 2]
        current_node,      # [B, K]  int64
        lengths,           # [B, K]  float32
        visited,           # [B, N+2] uint8
    ):
        B  = node_embeddings.size(0)
        K  = self.K
        D  = self.embed_dim
        N2 = node_embeddings.size(1)

        visited_bool = visited.float().bool()

        mask_list   = []
        query_parts = []

        for k in range(K):
            cur_k = current_node[:, k:k+1]  # [B, 1]

            # Current coordinate of agent k: [B, 2]
            cur_coord = torch.gather(
                coords, 1, cur_k[:, :, None].expand(B, 1, 2)
            ).squeeze(1)

            # Distance from current position to every node: [B, N2]
            dist_to_all = (coords - cur_coord[:, None, :]).norm(p=2, dim=-1)

            # Exceeds remaining budget
            exceeds = lengths[:, k:k+1] + dist_to_all > node_max_length

            # Already at end depot
            in_end = (cur_k == N2 - 1).expand(B, N2)

            # Combined infeasibility mask (1 = cannot visit)
            mask = visited_bool | exceeds | in_end
            mask[:, 0]     = True   # block start depot
            mask[:, N2-1]  = False  # always allow end depot
            mask_list.append(mask[:, None, :])  # [B, 1, N2]

            # Step context (matches _get_parallel_step_context)
            emb_cur    = torch.gather(
                node_embeddings, 1, cur_k[:, :, None].expand(B, 1, D)
            )  # [B, 1, D]
            end_max    = node_max_length[:, -1:]   # [B, 1]
            start_max  = node_max_length[:, 0:1]   # [B, 1]
            rem_norm   = (end_max - lengths[:, k:k+1]) / (start_max + 1e-6)
            step_ctx   = torch.cat([emb_cur, rem_norm[:, :, None]], dim=-1)  # [B, 1, D+1]
            step_ctx_p = self.model.project_step_context(step_ctx)            # [B, 1, D]
            q_k        = (fixed_context + step_ctx_p)[:, :, None, :]         # [B, 1, 1, D]
            query_parts.append(q_k)

        query = torch.cat(query_parts, dim=2)  # [B, 1, K, D]

        log_p, _ = self.model._one_to_many_logits(
            query, glimpse_key, glimpse_val, logit_key, mask_list
        )
        log_p = torch.log_softmax(log_p[:, 0, :, :] / self.model.temp, dim=-1)
        return log_p  # [B, K, N+2]


# ──────────────────────────────────────────────────────────────────────────────

def load_marga(ckpt_path: str, num_agents: int) -> AttentionModel:
    args_path = os.path.join(os.path.dirname(ckpt_path), 'args.json')
    with open(args_path) as f:
        cfg = json.load(f)

    problem = load_problem('top')
    model   = AttentionModel(
        cfg['embedding_dim'], cfg['hidden_dim'], problem,
        n_encode_layers=cfg['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=cfg['normalization'],
        tanh_clipping=cfg['tanh_clipping'],
        checkpoint_encoder=False,
        shrink_size=None,
        num_depots=cfg.get('num_depots', 1),
        num_agents=num_agents,
        use_rope=cfg.get('use_rope', True),
        rope_theta=cfg.get('rope_theta', 1e4),
        use_entropy_scheduling=cfg.get('use_entropy_scheduling', True),
    )
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
    model.load_state_dict({**model.state_dict(), **state}, strict=False)
    model.set_decode_type('greedy')
    model.eval()
    print(f'  Loaded  embed_dim={cfg["embedding_dim"]}  K={num_agents}')
    return model


def export_encoder(model: AttentionModel, n: int, out_dir: str):
    print('\n[1/2] Exporting encoder.onnx ...')
    wrapper = EncoderWrapper(model).eval()
    B = 1
    dummy = (
        torch.randn(B, 2),
        torch.randn(B, n, 2),
        torch.ones(B, n),
        torch.full((B,), 2.0),
    )
    path = os.path.join(out_dir, 'encoder.onnx')
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, path,
            input_names  = ['depot', 'loc', 'prize', 'max_length'],
            output_names = ['node_embeddings', 'fixed_context',
                            'glimpse_key', 'glimpse_val', 'logit_key',
                            'node_max_length', 'coords'],
            dynamic_axes = {
                'depot':           {0: 'B'},
                'loc':             {0: 'B', 1: 'N'},
                'prize':           {0: 'B', 1: 'N'},
                'max_length':      {0: 'B'},
                'node_embeddings': {0: 'B', 1: 'N2'},
                'fixed_context':   {0: 'B'},
                'glimpse_key':     {1: 'B', 3: 'N2'},
                'glimpse_val':     {1: 'B', 3: 'N2'},
                'logit_key':       {0: 'B', 2: 'N2'},
                'node_max_length': {0: 'B', 1: 'N2'},
                'coords':          {0: 'B', 1: 'N2'},
            },
            opset_version=17,
        )
    import onnx
    onnx.save_model(onnx.load(path), path, save_as_external_data=False)
    print(f'  → {path}  ({os.path.getsize(path) / 1024**2:.1f} MB)')


def export_decoder_step(model: AttentionModel, n: int, out_dir: str):
    print('\n[2/2] Exporting decoder_step.onnx ...')
    wrapper = DecoderStepWrapper(model).eval()
    K, D, H = model.num_agents, model.embedding_dim, model.n_heads
    dh = D // H
    N2 = n + 2
    B  = 1
    dummy = (
        torch.randn(B, N2, D),
        torch.randn(B, 1,  D),
        torch.randn(H, B,  1, N2, dh),
        torch.randn(H, B,  1, N2, dh),
        torch.randn(B, 1,  N2, D),
        torch.randn(B, N2),
        torch.randn(B, N2, 2),
        torch.zeros(B, K,  dtype=torch.int64),
        torch.zeros(B, K),
        torch.zeros(B, N2, dtype=torch.uint8),
    )
    path = os.path.join(out_dir, 'decoder_step.onnx')
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, path,
            input_names  = ['node_embeddings', 'fixed_context',
                            'glimpse_key', 'glimpse_val', 'logit_key',
                            'node_max_length', 'coords',
                            'current_node', 'lengths', 'visited'],
            output_names = ['log_p'],
            dynamic_axes = {
                'node_embeddings': {0: 'B'},
                'fixed_context':   {0: 'B'},
                'glimpse_key':     {1: 'B'},
                'glimpse_val':     {1: 'B'},
                'logit_key':       {0: 'B'},
                'node_max_length': {0: 'B'},
                'coords':          {0: 'B'},
                'current_node':    {0: 'B'},
                'lengths':         {0: 'B'},
                'visited':         {0: 'B'},
                'log_p':           {0: 'B'},
            },
            opset_version=17,
        )
    import onnx
    onnx.save_model(onnx.load(path), path, save_as_external_data=False)
    print(f'  → {path}  ({os.path.getsize(path) / 1024**2:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(
        description='Export MARGA to ONNX for Android deployment'
    )
    parser.add_argument(
        '--ckpt',
        default=os.path.join(
            ROOT, 'pretrained/Mixed-training/top_mixed100/MARGA/epoch-99.pt'
        ),
        help='Path to model checkpoint (.pt)',
    )
    parser.add_argument('--n_nodes',    type=int, default=100,
                        help='Number of customer nodes (default: 100)')
    parser.add_argument('--num_agents', type=int, default=5,
                        help='Number of agents / routes (default: 5)')
    parser.add_argument(
        '--output',
        default=os.path.join(ROOT, 'deploy/exported'),
        help='Output directory for .onnx files and config.json',
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading checkpoint: {args.ckpt}')
    model = load_marga(args.ckpt, args.num_agents)

    export_encoder(model, args.n_nodes, args.output)
    export_decoder_step(model, args.n_nodes, args.output)

    cfg = {
        'n_nodes':    args.n_nodes,
        'num_agents': args.num_agents,
        'embed_dim':  model.embedding_dim,
        'n_heads':    model.n_heads,
        'tanh_clip':  model.tanh_clipping,
        'n2':         args.n_nodes + 2,
    }
    cfg_path = os.path.join(args.output, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f'\n  Config → {cfg_path}')
    print('\n=== Export complete ===')
    print(f'  Files in {args.output}:')
    for fn in ['encoder.onnx', 'decoder_step.onnx', 'config.json']:
        fp = os.path.join(args.output, fn)
        if os.path.exists(fp):
            print(f'    {fn}  ({os.path.getsize(fp) / 1024**2:.2f} MB)')
    print('\nNext step: copy these 3 files to your Android project assets/ folder.')


if __name__ == '__main__':
    main()

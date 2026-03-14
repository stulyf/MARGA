#!/usr/bin/env python

import os
import json
import torch
import pprint as pp
import torch.optim as optim

try:
    from tensorboard_logger import Logger as TbLogger  # pip install protobuf==3.20.*
    _TB_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    TbLogger = None
    _TB_IMPORT_ERROR = exc

from utils.reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from utils.train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
from utils.options import get_options


def _parse_val_dataset_map(mix_dists, entries):
    """Parse --val_datasets argument into a dist->path mapping."""
    if not mix_dists:
        return {}

    mapping = {}
    entries = entries or []
    positional = []

    for item in entries:
        if '=' in item:
            dist, path = item.split('=', 1)
            dist = dist.strip()
            if dist:
                mapping[dist] = path.strip()
        else:
            positional.append(item.strip())

    if positional:
        if len(positional) != len(mix_dists):
            raise ValueError(
                "Number of positional --val_datasets entries must match --mix_dists"
            )
        for dist, path in zip(mix_dists, positional):
            mapping[dist] = path

    return {dist: mapping.get(dist) for dist in mix_dists if mapping.get(dist)}


def run(opts):

    # Save arguments so exact configuration can always be found
    if not opts.eval_only:
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

        # Optionally configure tensorboard
        tb_logger = None
        tb_dir = os.path.join(opts.save_dir, 'log_dir')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        if not opts.no_tensorboard:
            if TbLogger is None:
                raise RuntimeError(
                    "tensorboard_logger is unavailable; rerun with --no_tensorboard or install a compatible version"
                ) from _TB_IMPORT_ERROR
            tb_logger = TbLogger(tb_dir)

    # Print and save the run args
    pp.pprint(vars(opts))
    save_opts = open(os.path.join(opts.save_dir, 'log_dir', 'options_{}.txt'.format(opts.time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()

    # Set device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)
    mix_dists = [d.strip() for d in (opts.mix_dists or [])]
    opts.mix_dists = mix_dists if mix_dists else None

    # Load data
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data  = torch_load_cpu(load_path)

    # Load model
    assert opts.model == 'attention', "Only the attention model is supported"
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_depots=opts.num_depots,
        num_agents=opts.num_agents,
        use_rope=opts.use_rope,
        rope_theta=opts.rope_theta,
        use_entropy_scheduling=opts.use_entropy_scheduling
    ).to(opts.device)

    # Multi-GPU
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Load baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Load optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Load learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Load validation data (support multiple distributions for mixed training)
    if mix_dists:
        val_dataset = {}
        val_map = _parse_val_dataset_map(mix_dists, opts.val_datasets)
        if opts.val_dataset and not val_map:
            val_map = {dist: opts.val_dataset for dist in mix_dists}

        for dist in mix_dists:
            path = val_map.get(dist)
            if path is not None:
                if not os.path.isfile(path):
                    print(f"[val] Warning: dataset for '{dist}' not found at {path}, generating on the fly.")
                    path = None
                else:
                    print(f"[val] Using validation dataset for distribution '{dist}': {path}")
                    val_dataset[dist] = problem.make_dataset(
                        size=opts.graph_size,
                        num_samples=opts.val_size,
                        filename=path,
                        distribution=dist,
                        num_agents=opts.num_agents,
                        num_depots=opts.num_depots,
                        max_length=opts.max_length
                    )
            if dist not in val_dataset:
                print(f"[val] No dataset path provided for distribution '{dist}', generating on the fly.")
                val_dataset[dist] = problem.make_dataset(
                    size=opts.graph_size,
                    num_samples=opts.val_size,
                    distribution=dist,
                    num_agents=opts.num_agents,
                    num_depots=opts.num_depots,
                    max_length=opts.max_length
                )
    else:
        val_dataset = problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.val_size,
            filename=opts.val_dataset,
            distribution=opts.data_distribution,
            num_agents=opts.num_agents,
            num_depots=opts.num_depots,
            max_length=opts.max_length
        )

    # Resume training
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

        # Set the random states. Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # Evaluate validation data
    if opts.eval_only:
        if isinstance(val_dataset, dict):
            for dist, ds in val_dataset.items():
                print(f"[Eval] Distribution '{dist}'")
                validate(model, ds, opts)
        else:
            validate(model, val_dataset, opts)

    # Train
    else:
        for epoch in range(opts.initial_epoch, opts.initial_epoch + opts.epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())

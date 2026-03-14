import os
import time
import math
import torch
from tqdm import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={}".format(epoch, optimizer.param_groups[0]['lr']))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # Support mixed-distribution round-robin training if opts.mix_dists is provided.
    mix_dists = [d.strip() for d in (opts.mix_dists or [])]
    use_mixed = len(mix_dists) > 0
    if use_mixed:
        loaders = {}
        iters = {}
        for dist in mix_dists:
            ds = baseline.wrap_dataset(
                problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size,
                                     distribution=dist, num_agents=opts.num_agents,
                                     num_depots=opts.num_depots, max_length=opts.max_length)
            )
            loaders[dist] = DataLoader(ds, batch_size=opts.batch_size, num_workers=opts.num_workers)
            iters[dist] = iter(loaders[dist])
        steps_total = opts.epoch_size // opts.batch_size
    else:
        training_dataset = baseline.wrap_dataset(
            problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size,
                                 distribution=opts.data_distribution, num_agents=opts.num_agents,
                                 num_depots=opts.num_depots, max_length=opts.max_length)
        )
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    # Train one epoch
    if use_mixed:
        for i in tqdm(range(steps_total), disable=opts.no_progress_bar):
            dist = mix_dists[i % len(mix_dists)]
            try:
                batch = next(iters[dist])
            except StopIteration:
                iters[dist] = iter(loaders[dist])
                batch = next(iters[dist])
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                i,
                step,
                batch,
                tb_logger,
                opts
            )
            step += 1
    else:
        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )
            step += 1
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save trained model
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # Validate: multi-distribution if requested, else single
    if use_mixed:
        val_sets = val_dataset if isinstance(val_dataset, dict) else {}
        val_results = {}
        for dist in mix_dists:
            ds = None
            if isinstance(val_sets, dict):
                ds = val_sets.get(dist)
            if ds is None:
                ds = problem.make_dataset(
                    size=opts.graph_size,
                    num_samples=opts.val_size,
                    distribution=dist,
                    num_agents=opts.num_agents,
                    num_depots=opts.num_depots,
                    max_length=opts.max_length
                )
                if isinstance(val_sets, dict):
                    val_sets[dist] = ds
            val_results[dist] = validate(model, ds, opts)
        mixed = sum(val_results.values()) / len(val_results) if val_results else 0
        if not opts.no_tensorboard and tb_logger is not None:
            for k, v in val_results.items():
                tb_logger.log_value(f'val_avg_reward_{k}', v, step)
            tb_logger.log_value('val_avg_reward_mixed', mixed, step)
    else:
        avg_reward = validate(model, val_dataset, opts)
        if not opts.no_tensorboard and tb_logger is not None:
            tb_logger.log_value('val_avg_reward', avg_reward, step)

    # Update callback
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

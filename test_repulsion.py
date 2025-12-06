import argparse
import os
import os.path as osp
from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, Dataset

from train import (
    build_datasets,
    worker_init_fn,
    init_model,
    gen_answer,
)
from plot_timesTwo_energy import compute_energy_grid, plot_and_save


class TimesTwoListDataset(Dataset):
    """
    Simple dataset for timesTwo when testing on a fixed list of inputs.
    """

    def __init__(self, xs):
        xs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
        self.inp = xs
        self.out = 2.0 * xs
        self.inp_dim = 1
        self.out_dim = 1

    def __len__(self):
        return self.inp.size(0)

    def __getitem__(self, idx):
        return self.inp[idx], self.out[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-time repulsion evaluation for EBM models"
    )
    # Checkpoint / experiment config
    parser.add_argument(
        "--dataset",
        type=str,
        default="timesTwo",
        help="Dataset name to use for testing (default: timesTwo).",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="default",
        help="Experiment name used during training (to locate checkpoint).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.pth). "
             "If not provided, defaults to result/<exp>/model_latest.pth",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for test-time evaluation.",
    )
    parser.add_argument(
        "--data_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=50,
        help="Maximum number of test batches to evaluate (for speed). "
             "-1 evaluates the full test set.",
    )

    # Repulsion hyperparameters
    parser.add_argument(
        "--lambda_repulsion",
        type=float,
        default=5.0,
        help="Weight of the repulsive energy term.",
    )
    parser.add_argument(
        "--alpha_amp",
        type=float,
        default=1.0,
        help="Base coefficient for amplitude schedule A(err).",
    )
    parser.add_argument(
        "--p_amp",
        type=float,
        default=1.0,
        help="Power for amplitude schedule A(err) = alpha_amp * err ** p_amp.",
    )
    parser.add_argument(
        "--sigma_base",
        type=float,
        default=1,
        help="Base standard deviation for Gaussian kernels in solution space.",
    )
    parser.add_argument(
        "--beta_sigma",
        type=float,
        default=0.0,
        help="Controls how kernel width changes with error: "
             "sigma(err) = sigma_base / (1 + beta_sigma * err).",
    )
    parser.add_argument(
        "--num_iters_per_sample",
        type=int,
        default=3,
        help="Number of repulsion refinement iterations per test sample.",
    )
    parser.add_argument(
        "--refine_steps",
        type=int,
        default=100,
        help="Number of gradient steps for each repulsion-based proposal.",
    )
    parser.add_argument(
        "--memory_mode",
        type=str,
        default="local",
        choices=["none", "local", "global", "both"],
        help="Type of memory to use for repulsion.",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=1000,
        help="Maximum number of proposals stored in global memory.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help="Comma-separated list of scalar inputs x for timesTwo testing.",
    )
    parser.add_argument(
        "--inputs_file",
        type=str,
        default=None,
        help="Path to a text file with one scalar x per line for timesTwo testing.",
    )
    parser.add_argument(
        "--plot_combined",
        action="store_true",
        help="If set, plots combined (base + repulsion) energy landscape "
             "for timesTwo after each proposal.",
    )

    return parser.parse_args()


def load_checkpoint_and_flags(ckpt_path, device, batch_size, data_workers, dataset_name=None):
    """
    Load a training checkpoint and adapt FLAGS for evaluation.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_FLAGS = checkpoint["FLAGS"]

    # Shallow copy into a simple namespace so we can override a few fields
    # without affecting the original checkpoint object.
    FLAGS = SimpleNamespace(**vars(ckpt_FLAGS))
    if dataset_name is not None:
        FLAGS.dataset = dataset_name
    FLAGS.batch_size = batch_size
    FLAGS.data_workers = data_workers
    # Ensure we are not in training mode and do not rely on replay buffer
    FLAGS.train = False
    FLAGS.no_replay_buffer = True
    FLAGS.replay_buffer = False

    # Build datasets & dataloaders using the training configuration
    dataset, test_dataset = build_datasets(FLAGS)
    test_loader = DataLoader(
        test_dataset,
        num_workers=FLAGS.data_workers,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    # Instantiate model and load weights
    model, _ = init_model(FLAGS, device, dataset)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, FLAGS, test_loader


def amplitude_from_error(err, alpha_amp, p_amp, max_amp=10.0):
    amp = alpha_amp * (err ** p_amp)
    return torch.clamp(amp, max=max_amp)


def sigma_from_error(err, sigma_base, beta_sigma, min_sigma=1e-3, max_sigma=10.0):
    # sigma = sigma_base / (1.0 + beta_sigma * err)
    sigma = sigma_base*(err**0.5)*0.2
    return torch.clamp(sigma, min=min_sigma, max=max_sigma)


def compute_repulsion_energy(pred, memory, args, device):
    """
    Compute R(z; M) for a batch of joint points `pred` and a memory of
    past (z_k, err_k) entries, where z = (y, x) for timesTwo.

    pred: (B, D) joint coordinates (e.g., [y, x] for D=2)
    memory: list of dicts with keys 'joint' (tensor of shape (D,)),
            'err' (float), and optional 'weight' (float).
    """
    if len(memory) == 0 or args.lambda_repulsion == 0.0:
        return torch.zeros(pred.size(0), device=device)

    # Stack memory entries for vectorised computation
    joints = torch.stack([m["joint"].to(device) for m in memory], dim=0)  # (M, D)
    errs = torch.tensor([m["err"] for m in memory], device=device)  # (M,)
    weights = torch.tensor(
        [m.get("weight", 1.0) for m in memory], device=device
    )  # (M,)

    # Compute amplitude and sigma per memory item.
    # We use a mix of **relative** and **absolute** error so that:
    #   - near-zero absolute error  -> near-zero repulsion
    #   - relatively better (but not perfect) proposals still exert
    #     non-trivial repulsion.
    if errs.numel() > 1:
        err_min = errs.min()
        err_max = errs.max()
        denom = (err_max - err_min).clamp_min(1e-8)
        rel_errs = (errs - err_min) / denom  # in [0, 1]
    else:
        rel_errs = torch.ones_like(errs)

    # Raise the floor on relative error slightly so that "best so far"
    # points still contribute some repulsion, unless their absolute
    # error is essentially zero.
    rel_floor = 0.1
    rel_scaled = rel_floor + (1.0 - rel_floor) * rel_errs  # in [rel_floor, 1]

    # Gate by absolute error so that truly near-perfect solutions
    # do not repulse much.
    abs_tau = 1e-3
    abs_weight = errs / (errs + abs_tau)  # in [0, 1)

    eff_err = rel_scaled * abs_weight  # in [0, 1), ~0 when abs err ~0

    amps = amplitude_from_error(eff_err, args.alpha_amp, args.p_amp) * weights  # (M,)
    sigmas = sigma_from_error(errs, args.sigma_base, args.beta_sigma)  # (M,)

    # pred: (B, D), joints: (M, D) -> diff: (B, M, D)
    diff = pred.unsqueeze(1) - joints.unsqueeze(0)
    # Squared L2 norm along D
    sq_norm = (diff ** 2).sum(dim=-1)  # (B, M)

    # Gaussian kernels r_k(y) = A_k * exp(-0.5 * ||y - y_k||^2 / sigma_k^2)
    denom = 2.0 * (sigmas ** 2)  # (M,)
    # (B, M) broadcast with (M,) -> (B, M)
    exponents = -sq_norm / denom.unsqueeze(0)
    kernels = amps.unsqueeze(0) * torch.exp(exponents)  # (B, M)

    # Sum over memory
    repulsion = kernels.sum(dim=1)  # (B,)
    return repulsion


def compute_repulsion_grad(pred, x, memory, args, device):
    """
    Analytic gradient of R(z; M) wrt the y-coordinate for isotropic
    Gaussian kernels defined over the joint z = (y, x).

    pred: (B, 1) current y proposals
    x:    (B, 1) corresponding x inputs
    Returns gradient w.r.t. y with shape (B, 1).
    """
    if len(memory) == 0 or args.lambda_repulsion == 0.0:
        return torch.zeros_like(pred)

    joints_mem = torch.stack(
        [m["joint"].to(device) for m in memory], dim=0
    )  # (M, D)
    errs = torch.tensor([m["err"] for m in memory], device=device)  # (M,)
    weights = torch.tensor(
        [m.get("weight", 1.0) for m in memory], device=device
    )  # (M,)

    # Use the same relative/absolute-error scaling as in compute_repulsion_energy
    if errs.numel() > 1:
        err_min = errs.min()
        err_max = errs.max()
        denom = (err_max - err_min).clamp_min(1e-8)
        rel_errs = (errs - err_min) / denom
    else:
        rel_errs = torch.ones_like(errs)

    rel_floor = 0.1
    rel_scaled = rel_floor + (1.0 - rel_floor) * rel_errs

    abs_tau = 1e-3
    abs_weight = errs / (errs + abs_tau)

    eff_err = rel_scaled * abs_weight

    amps = amplitude_from_error(eff_err, args.alpha_amp, args.p_amp) * weights  # (M,)
    sigmas = sigma_from_error(errs, args.sigma_base, args.beta_sigma)  # (M,)

    # Form joint coordinates for current batch: z = (y, x)
    joint = torch.cat([pred, x], dim=-1)  # (B, D)
    diff = joint.unsqueeze(1) - joints_mem.unsqueeze(0)  # (B, M, D)
    sq_norm = (diff ** 2).sum(dim=-1)  # (B, M)

    denom = 2.0 * (sigmas ** 2)  # (M,)
    exponents = -sq_norm / denom.unsqueeze(0)
    kernels = amps.unsqueeze(0) * torch.exp(exponents)  # (B, M)

    # d/dy r_k(z) = r_k(z) * (-(y - y_k) / sigma_k^2)
    factor = -1.0 / (sigmas ** 2)  # (M,)
    diff_y = diff[..., 0]  # (B, M) assuming joint = [y, x, ...]
    grad_y = kernels * diff_y * factor.unsqueeze(0)  # (B, M)
    grad_y = grad_y.sum(dim=1, keepdim=True)  # (B, 1)
    return grad_y


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean(dim=-1)


def update_memory(memory, new_entries, max_size):
    """
    Append new entries to memory, pruning to at most max_size elements.
    Each entry is a dict with keys:
      - 'joint': joint coordinates tensor (e.g., [y, x] for timesTwo)
      - 'err': scalar error
      - optional 'weight'
      - optional 'x', 'y' for convenience / debugging
    """
    memory.extend(new_entries)
    if max_size > 0 and len(memory) > max_size:
        # Simple FIFO pruning
        overflow = len(memory) - max_size
        del memory[0:overflow]


def optimize_prediction_under_energy(model, FLAGS, inp, target, num_steps,
                                     memory, args, device):
    """
    Simple gradient-descent optimizer that mirrors the core logic in
    `train.py` for timesTwo: we treat `model` as an energy network
    taking [pred, inp] and perform gradient descent on pred.

    If `memory` is provided, we add a repulsive term R((y, x); memory)
    to the base energy.
    """
    if getattr(FLAGS, "mem", False):
        raise NotImplementedError("mem-based models are not supported in test_repulsion.")

    # Random initialization in output space, as in train.py
    pred = (torch.rand_like(target) - 0.5) * 10

    step_lr = getattr(FLAGS, "step_lr", 100.0)

    for _ in range(num_steps):
        pred = pred.detach().clone().requires_grad_(True)

        # Base energy E_theta(x, y)
        im_merge = torch.cat([pred, inp], dim=-1)
        base_energy_vec = model(im_merge).squeeze(-1)  # (B,)
        total_energy_vec = base_energy_vec

        # Optional repulsion in joint (y, x) space
        if memory is not None and len(memory) > 0 and args.lambda_repulsion != 0.0:
            joints = torch.cat([pred, inp], dim=-1)  # (B, D)
            rep_term = compute_repulsion_energy(joints, memory, args, device)  # (B,)
            total_energy_vec = total_energy_vec + args.lambda_repulsion * rep_term

        total = total_energy_vec.sum()
        grad_pred, = torch.autograd.grad([total], [pred], create_graph=False)

        pred = pred - step_lr * grad_pred

    return pred.detach()


def combine_batch_plots(exp_name, batch_idx, num_iters_per_sample):
    """
    Combine the per-batch PNGs (baseline + combined iterations) into
    a single horizontal panel for easier visual comparison.
    """
    plot_dir = osp.join("result", exp_name, "repulsion_plots")

    if not osp.exists(plot_dir):
        return

    images_info = []

    # Baseline energy plot
    baseline_path = osp.join(
        plot_dir,
        f"baseline_energy_b{batch_idx:04d}_it-1.png",
    )
    if osp.exists(baseline_path):
        images_info.append(("baseline", baseline_path))

    # Combined energy plots: iter 0 (after baseline proposal) up to num_iters_per_sample
    for it in range(num_iters_per_sample + 1):
        combined_path = osp.join(
            plot_dir,
            f"combined_energy_b{batch_idx:04d}_it{it:02d}.png",
        )
        if osp.exists(combined_path):
            images_info.append((f"iter {it}", combined_path))

    if not images_info:
        return

    imgs = [mpimg.imread(path) for _, path in images_info]

    n_cols = len(imgs)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4 * n_cols, 4),
        dpi=150,
    )
    if n_cols == 1:
        axes = [axes]

    for ax, (label, _), img in zip(axes, images_info, imgs):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    out_path = osp.join(
        plot_dir,
        f"combined_panel_b{batch_idx:04d}.png",
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


class BaseTimesTwoEnergy(torch.nn.Module):
    """
    Wrapper model that exposes the base EBM energy on the (x, y) grid
    used for timesTwo visualizations, without any repulsion term.
    """

    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device

    def forward(self, pairs):
        # pairs: (N, 2) with [y, x] as in plot_timesTwo_energy.py.
        # To match the original training visualization exactly, we call the
        # underlying model on [y, x] directly, without adding scratchpads.
        energy = self.base_model(pairs.to(self.device))
        return energy.view(-1)


class CombinedTimesTwoEnergy(torch.nn.Module):
    """
    Wrapper model that adds repulsion energy on top of the base EBM
    for visualization on the (x, y) grid (timesTwo only).
    """

    def __init__(self, base_model, args, device, memory):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.device = device
        self.memory = memory

    def forward(self, pairs):
        # pairs: (N, 2) with [y, x] as in plot_timesTwo_energy.py
        # Match the base energy computation used in the original training
        # visualization by calling the model directly on [y, x].
        pairs = pairs.to(self.device)
        base_energy = self.base_model(pairs).view(-1)
        # Repulsion in the joint (y, x) plane, centred at proposal points.
        repulsion = compute_repulsion_energy(pairs, self.memory, self.args, self.device)
        return base_energy + self.args.lambda_repulsion * repulsion


def get_effective_memory(local_memory, global_memory, args):
    mem = []
    if args.memory_mode in ("local", "both"):
        mem.extend(local_memory)
    if args.memory_mode in ("global", "both"):
        mem.extend(global_memory)
    return mem


def maybe_plot_combined_landscape(model, args, device, memory, batch_idx, iter_idx, proposals=None):
    """
    For timesTwo, plot combined (base + repulsion) energy after each proposal.
    """
    if not getattr(args, "plot_combined", False):
        return

    if getattr(model.F, "dataset", None) != "timesTwo":
        return

    # If no repulsion is active, just visualize base model via wrapper anyway.
    effective_memory = memory or []

    plot_dir = osp.join("result", getattr(model.F, "exp", "default"), "repulsion_plots")
    os.makedirs(plot_dir, exist_ok=True)

    combined_model = CombinedTimesTwoEnergy(model, args, device, effective_memory)

    # Match the plotting ranges used in train.py for timesTwo
    scale = 10.0
    x_min, x_max = -scale, scale
    y_min, y_max = -2 * scale, 2 * scale
    resolution = 200

    with torch.no_grad():
        XX, YY, energy = compute_energy_grid(
            combined_model,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            device=device,
        )

    # Prepare proposal coordinates (up to current iteration) for overlay.
    proposal_points = None
    if proposals is not None:
        xs = []
        ys = []
        iters = []
        for p in proposals:
            step = p.get("iter", 0)
            if step <= iter_idx:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
                iters.append(float(step))
        if xs:
            import numpy as np  # local import to avoid extra top-level dependency

            proposal_points = {
                "xs": np.array(xs, dtype=np.float32),
                "ys": np.array(ys, dtype=np.float32),
                "iters": np.array(iters, dtype=np.float32),
            }

    out_path = osp.join(
        plot_dir,
        f"combined_energy_b{batch_idx:04d}_it{iter_idx:02d}.png",
    )
    title = f"Combined E (batch {batch_idx}, it {iter_idx})"
    plot_and_save(
        XX,
        YY,
        energy,
        out_path,
        title,
        plot_argmin=True,
        plot_grad=False,
        plot_grad2=False,
        proposal_points=proposal_points,
    )


def maybe_plot_baseline_landscape(model, args, device, batch_idx, proposals=None):
    """
    For timesTwo, plot the pure baseline energy landscape (no repulsion),
    using the same coordinate system as the combined plots.
    """
    if not getattr(args, "plot_combined", False):
        return

    if getattr(model.F, "dataset", None) != "timesTwo":
        return

    plot_dir = osp.join("result", getattr(model.F, "exp", "default"), "repulsion_plots")
    os.makedirs(plot_dir, exist_ok=True)

    baseline_model = BaseTimesTwoEnergy(model, device)

    scale = 10.0
    x_min, x_max = -scale, scale
    y_min, y_max = -2 * scale, 2 * scale
    resolution = 200

    with torch.no_grad():
        XX, YY, energy = compute_energy_grid(
            baseline_model,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            device=device,
        )

    # For the baseline plot, only show baseline proposals (iter == 0).
    proposal_points = None
    if proposals is not None:
        xs = []
        ys = []
        iters = []
        for p in proposals:
            step = p.get("iter", 0)
            if step == 0:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
                iters.append(float(step))
        if xs:
            import numpy as np  # local import to avoid extra top-level dependency

            proposal_points = {
                "xs": np.array(xs, dtype=np.float32),
                "ys": np.array(ys, dtype=np.float32),
                "iters": np.array(iters, dtype=np.float32),
            }

    out_path = osp.join(
        plot_dir,
        f"baseline_energy_b{batch_idx:04d}_it-1.png",
    )
    title = f"Baseline E (batch {batch_idx}, it -1)"
    plot_and_save(
        XX,
        YY,
        energy,
        out_path,
        title,
        plot_argmin=True,
        plot_grad=False,
        plot_grad2=False,
        proposal_points=proposal_points,
    )


def eval_with_repulsion(model, test_loader, args, device):
    """
    Run baseline and repulsion-augmented evaluation.
    """
    model.eval()
    global_memory = []

    baseline_errs = []
    repulsion_best_errs = []

    num_batches_done = 0

    for batch_idx, (inp, target) in enumerate(test_loader):
        if args.max_batches > 0 and num_batches_done >= args.max_batches:
            break

        inp = inp.float().to(device)
        target = target.float().to(device)

        # Baseline: mirror the core gradient-descent logic from train.py
        # to obtain a proposal under the learned energy E_theta(x, y).
        pred_base = optimize_prediction_under_energy(
            model=model,
            FLAGS=model.F,
            inp=inp,
            target=target,
            num_steps=80,  # match test-time steps in train.py
            memory=None,
            args=args,
            device=device,
        )
        base_err = mse_loss(pred_base, target)  # (B,)
        baseline_errs.append(base_err.cpu())

        # Track all proposals in this batch for visualization
        all_proposals = []

        # Repulsion-enhanced inference, local memory per batch
        local_memory = []
        best_err = base_err.clone()
        best_pred = pred_base.clone()

        # Seed memory with baseline proposal if using repulsion
        base_entries = []
        for i in range(pred_base.size(0)):
            y_i = pred_base[i].detach().cpu()
            x_i = inp[i].detach().cpu()
            all_proposals.append(
                {
                    "x": x_i.item(),
                    "y": y_i.item(),
                    "iter": 0,  # baseline proposal
                }
            )
            base_entries.append(
                {
                    "joint": torch.cat([y_i, x_i], dim=-1),
                    "y": y_i,
                    "x": x_i,
                    "err": float(base_err[i].item()),
                }
            )
        if args.memory_mode in ("local", "both"):
            update_memory(local_memory, base_entries, max_size=args.memory_size)
        if args.memory_mode in ("global", "both"):
            update_memory(global_memory, base_entries, max_size=args.memory_size)

        # Plot pure baseline landscape (no repulsion) with baseline proposals
        maybe_plot_baseline_landscape(
            model,
            args,
            device,
            batch_idx=batch_idx,
            proposals=all_proposals,
        )

        # Optionally visualize combined landscape after baseline proposal (iter 0)
        effective_memory = get_effective_memory(local_memory, global_memory, args)
        maybe_plot_combined_landscape(
            model,
            args,
            device,
            effective_memory,
            batch_idx=batch_idx,
            iter_idx=0,
            proposals=all_proposals,
        )

        # Repulsion-based proposals: at each refinement iteration, we
        # sample a NEW proposal from scratch (random init) and run the
        # same simple gradient-descent logic as training, but on the
        # combined energy:
        #   E'(x, y) = E_theta(x, y) + lambda_repulsion * R((y, x); M_t).
        for t in range(args.num_iters_per_sample):
            # Snapshot current memory for this refinement pass
            effective_memory = get_effective_memory(local_memory, global_memory, args)

            # Optimize under E' from scratch
            pred = optimize_prediction_under_energy(
                model=model,
                FLAGS=model.F,
                inp=inp,
                target=target,
                num_steps=args.refine_steps,
                memory=effective_memory,
                args=args,
                device=device,
            )

            # After inner loop, evaluate error under current repulsive energy
            err = mse_loss(pred, target)  # (B,)

            # Track best error so far
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_pred = torch.where(
                improved.unsqueeze(-1), pred, best_pred
            )

            # Record this proposal for visualization
            for i in range(pred.size(0)):
                all_proposals.append(
                    {
                        "x": float(inp[i].item()),
                        "y": float(pred[i].item()),
                        "iter": t + 1,  # proposal from refinement step t
                    }
                )

            # Plot combined landscape under the SAME memory that was used
            # to generate this proposal (without including this proposal
            # yet in the repulsion term).
            effective_memory = get_effective_memory(local_memory, global_memory, args)
            maybe_plot_combined_landscape(
                model,
                args,
                device,
                effective_memory,
                batch_idx=batch_idx,
                iter_idx=t + 1,
                proposals=all_proposals,
            )

            # Now update the memory with this new proposal so that the
            # NEXT refinement step sees an updated repulsive landscape.
            new_entries = []
            for i in range(pred.size(0)):
                y_i = pred[i].detach().cpu()
                x_i = inp[i].detach().cpu()
                entry = {
                    "joint": torch.cat([y_i, x_i], dim=-1),
                    "y": y_i,
                    "x": x_i,
                    "err": float(err[i].item()),
                }
                new_entries.append(entry)

            if args.memory_mode in ("local", "both"):
                update_memory(local_memory, new_entries, max_size=args.memory_size)
            if args.memory_mode in ("global", "both"):
                update_memory(global_memory, new_entries, max_size=args.memory_size)

        repulsion_best_errs.append(best_err.detach().cpu())

        # After generating all individual plots for this batch, optionally
        # create a single combined panel image for easier inspection.
        if getattr(args, "plot_combined", False):
            exp_name = getattr(model.F, "exp", "default")
            combine_batch_plots(exp_name, batch_idx, args.num_iters_per_sample)

        num_batches_done += 1

    baseline_errs = torch.cat(baseline_errs, dim=0)
    repulsion_best_errs = torch.cat(repulsion_best_errs, dim=0)

    results = {
        "baseline_mse_mean": baseline_errs.mean().item(),
        "baseline_mse_std": baseline_errs.std(unbiased=False).item(),
        "repulsion_mse_mean": repulsion_best_errs.mean().item(),
        "repulsion_mse_std": repulsion_best_errs.std(unbiased=False).item(),
    }
    return results


def main():
    args = parse_args()

    if args.ckpt is None:
        ckpt_dir = osp.join("result", args.exp)
        args.ckpt = osp.join(ckpt_dir, "model_latest.pth")

    if not osp.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}")

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, FLAGS, test_loader = load_checkpoint_and_flags(
        args.ckpt,
        device,
        batch_size=args.batch_size,
        data_workers=args.data_workers,
        dataset_name=args.dataset,
    )

    # Optionally override test loader with explicit timesTwo inputs
    input_values = None
    if args.inputs is not None and args.inputs_file is not None:
        raise ValueError("Specify at most one of --inputs or --inputs_file.")
    if args.inputs is not None:
        input_values = [
            float(s) for s in args.inputs.split(",") if s.strip() != ""
        ]
    elif args.inputs_file is not None:
        with open(args.inputs_file, "r") as f:
            input_values = [
                float(line.strip()) for line in f if line.strip() != ""
            ]

    if input_values is not None:
        if getattr(FLAGS, "dataset", None) != "timesTwo":
            raise ValueError(
                "Explicit input list is currently supported only for the timesTwo dataset."
            )
        custom_ds = TimesTwoListDataset(input_values)
        test_loader = DataLoader(
            custom_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # Attach FLAGS to model so that we can reuse a few hyperparameters (e.g., step_lr)
    model.F = FLAGS

    print("Running baseline vs. repulsion-augmented evaluation...")
    results = eval_with_repulsion(model, test_loader, args, device)

    print("=== Test-time Repulsion Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()



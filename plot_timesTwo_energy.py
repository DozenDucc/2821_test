import os
import os.path as osp
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import EBM
from dataset import timesTwo


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    flags = checkpoint.get('FLAGS', None)

    # Fallbacks if attributes are absent
    ood = getattr(flags, 'ood', False) if flags is not None else False
    mem = getattr(flags, 'mem', False) if flags is not None else False

    # For timesTwo, inp_dim = out_dim = 1
    ds = timesTwo('test', ood)
    model = EBM(ds.inp_dim, ds.out_dim, mem)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    return model, flags


def compute_energy_grid(model, x_min, x_max, y_min, y_max, resolution, device):
    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # shape: (res, res)

    # Model expects [pred(y), inp(x)] concatenated (see train.py gen_answer: torch.cat([pred, inp], -1))
    pairs = np.stack([YY.ravel(), XX.ravel()], axis=1).astype(np.float32)  # (N, 2) with order [y, x]
    pairs_t = torch.from_numpy(pairs).to(device)

    with torch.no_grad():
        energy = model(pairs_t).view(resolution, resolution).detach().cpu().numpy()

    return XX, YY, energy


def plot_and_save(XX, YY, energy, out_path, title, plot_argmin=False, plot_grad=False, plot_grad2=False):
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    xs_axis = XX[0, :]
    ys_axis = YY[:, 0]

    if plot_grad or plot_grad2:
        # Compute finite-difference derivatives on the grid as requested
        # np.gradient with coordinates returns [dE/dy, dE/dx]
        dE_dy = None
        d2E_dy2 = None
        if plot_grad or plot_grad2:
            dE_dy, _ = np.gradient(energy, ys_axis, xs_axis, edge_order=2)
        if plot_grad2:
            # Second derivative: derivative of dE/dy with respect to y
            d2E_dy2, _ = np.gradient(dE_dy, ys_axis, xs_axis, edge_order=2)

        ncols = 1 + (1 if plot_grad else 0) + (1 if plot_grad2 else 0)
        figsize = (6 * ncols, 5)
        fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=150, sharex=True, sharey=True)
        if ncols == 1:
            axes = [axes]

        # Energy subplot
        ax_idx = 0
        ax_energy = axes[ax_idx]
        im0 = ax_energy.imshow(energy, origin='lower', extent=extent, aspect='auto', cmap='viridis')
        fig.colorbar(im0, ax=ax_energy, label='Energy')
        ax_energy.set_xlabel('x')
        ax_energy.set_ylabel('y')
        ax_energy.set_title(title)

        # Optional overlays on energy pane
        x_vals = np.linspace(XX.min(), XX.max(), 200)
        y_vals = 2.0 * x_vals
        ax_energy.plot(x_vals, y_vals, color='white', linestyle='--', linewidth=1.0, alpha=0.8, label='y = 2x')
        if plot_argmin:
            argmin_indices = np.argmin(energy, axis=0)
            y_pred = ys_axis[argmin_indices]
            ax_energy.plot(xs_axis, y_pred, color='red', linewidth=1.5, label='argmin_y E(x, y)')
        ax_energy.legend(loc='upper right')

        # Optional: Gradient subplot dE/dy
        if plot_grad:
            ax_idx += 1
            ax_grad = axes[ax_idx]
            im1 = ax_grad.imshow(dE_dy, origin='lower', extent=extent, aspect='auto', cmap='coolwarm')
            fig.colorbar(im1, ax=ax_grad, label='dE/dy')
            ax_grad.set_xlabel('x')
            ax_grad.set_ylabel('y')
            ax_grad.set_title('dE/dy')

        # Optional: Second derivative subplot d2E/dy2
        if plot_grad2:
            ax_idx += 1
            ax_grad2 = axes[ax_idx]
            im2 = ax_grad2.imshow(d2E_dy2, origin='lower', extent=extent, aspect='auto', cmap='coolwarm')
            fig.colorbar(im2, ax=ax_grad2, label='d²E/dy²')
            ax_grad2.set_xlabel('x')
            ax_grad2.set_ylabel('y')
            ax_grad2.set_title('d²E/dy²')

        plt.tight_layout()
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return
    else:
        plt.figure(figsize=(6, 5), dpi=150)
        im = plt.imshow(energy, origin='lower', extent=extent, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Energy')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)

    # Optional: plot y = 2x guideline (timesTwo ground truth)
        x_vals = np.linspace(XX.min(), XX.max(), 200)
        y_vals = 2.0 * x_vals
        plt.plot(x_vals, y_vals, color='white', linestyle='--', linewidth=1.0, alpha=0.8, label='y = 2x')

    # Optional: overlay argmin_y E(x, y) curve (model's prediction per x)
        if plot_argmin:
            # energy shape: (num_ys, num_xs); argmin over y for each x (column-wise)
            argmin_indices = np.argmin(energy, axis=0)
            y_pred = ys_axis[argmin_indices]
            plt.plot(xs_axis, y_pred, color='red', linewidth=1.5, label='argmin_y E(x, y)')
        plt.legend(loc='upper right')

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timesTwo_energy_from_model(model, device, out_path,
                                    x_min=-2.5, x_max=2.5,
                                    y_min=-5.0, y_max=5.0,
                                    resolution=400,
                                    title='timesTwo EBM Energy',
                                    plot_argmin=False,
                                    plot_grad=False,
                                    plot_grad2=False):
    model_was_training = model.training
    model.eval()

    XX, YY, energy = compute_energy_grid(
        model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=resolution,
        device=device
    )

    plot_and_save(XX, YY, energy, out_path, title, plot_argmin=plot_argmin, plot_grad=plot_grad, plot_grad2=plot_grad2)

    if model_was_training:
        model.train()


def main():
    parser = argparse.ArgumentParser(description='Plot energy landscape for timesTwo EBM')
    parser.add_argument('--exp', type=str, default='timesTwo_experiment', help='experiment name (used for paths)')
    parser.add_argument('--logdir', type=str, default='cachedir', help='base log directory where checkpoints are saved')
    parser.add_argument('--ckpt', type=str, default='', help='explicit checkpoint path; overrides logdir/exp/model_latest.pth')
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')

    parser.add_argument('--x_min', type=float, default=-2.5, help='min x value')
    parser.add_argument('--x_max', type=float, default=2.5, help='max x value')
    parser.add_argument('--y_min', type=float, default=-5.0, help='min y value')
    parser.add_argument('--y_max', type=float, default=5.0, help='max y value')
    parser.add_argument('--resolution', type=int, default=400, help='grid resolution')

    parser.add_argument('--out', type=str, default='', help='output image path; defaults to result/<exp>/energy_heatmap.png')
    parser.add_argument('--plot_argmin', action='store_true', help='overlay argmin_y(E(x,y)) curve (model predictions) in red')
    parser.add_argument('--plot_grad', action='store_true', help='plot dE/dy landscape alongside the energy heatmap')
    parser.add_argument('--plot_grad2', action='store_true', help='plot d²E/dy² landscape alongside the energy heatmap')

    args = parser.parse_args()

    # Resolve paths
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = osp.join(args.logdir, args.exp, 'model_latest.pth')

    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found at: {ckpt_path}')

    if args.out:
        out_path = args.out
    else:
        out_path = osp.join('result', args.exp, 'energy_heatmap.png')

    # Device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load model
    model, flags = load_model(ckpt_path, device)

    # Compute grid
    XX, YY, energy = compute_energy_grid(
        model,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        resolution=args.resolution,
        device=device
    )

    # Plot
    title = 'timesTwo EBM Energy'
    if flags is not None and hasattr(flags, 'exp'):
        title += f' ({getattr(flags, "exp", "")})'

    plot_and_save(XX, YY, energy, out_path, title, plot_argmin=args.plot_argmin, plot_grad=args.plot_grad, plot_grad2=args.plot_grad2)
    print(f'Saved energy heatmap to: {out_path}')


if __name__ == '__main__':
    main()



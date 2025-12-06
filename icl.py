import os
import os.path as osp

import numpy as np
import torch

from train import build_datasets, init_model, gen_answer
from plot_timesTwo_energy import compute_energy_grid, plot_and_save


# Absolute path to the trained checkpoint you mentioned.
CKPT_PATH = "/home/ubuntu/2821_test/result/times-Two_0.0_experiment/model_latest.pth"

# Where to save the prediction + energy plot.
PLOT_PATH = "/home/ubuntu/2821_test/result/times-Two_0.0_experiment/icl_predictions.png"


def load_model_and_dataset(ckpt_path: str):
    """
    Load model + FLAGS from a training checkpoint, reusing the helpers
    from train.py to build the dataset and model.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    FLAGS = checkpoint["FLAGS"]

    use_cuda = getattr(FLAGS, "cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Reuse the exact dataset + model construction used during training.
    dataset, _ = build_datasets(FLAGS)
    model, _ = init_model(FLAGS, device, dataset)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, FLAGS, dataset, device


def run_inference(model, FLAGS, dataset, device, inputs, num_steps: int | None = None):
    """
    Run the same iterative inference logic as in train.test (gen_answer)
    on a user-specified list of inputs.
    """
    # Convert Python list -> tensor with correct input dimension.
    inp = torch.tensor(inputs, dtype=torch.float32).view(-1, dataset.inp_dim).to(device)

    # Random initial prediction and zero scratchpad, matching train.test.
    pred_init = (torch.rand(inp.size(0), dataset.out_dim, device=device) - 0.5) * 2
    scratch = torch.zeros_like(inp)

    if num_steps is None:
        # Default to FLAGS.num_steps if available, otherwise fall back to 80
        num_steps = 500

    # Use the exact iterative update from train.gen_answer.
    final_pred, preds, im_grads, energies, scratch, logits = gen_answer(
        inp, FLAGS, model, pred_init, scratch, num_steps
    )

    return inp.detach().cpu(), final_pred.detach().cpu()


def plot_predictions_on_energy(model, device, xs_cpu, ys_pred_cpu, out_path: str):
    """
    For timesTwo, reuse the existing energy plotting utilities by:
      1) computing the EBM energy grid via `compute_energy_grid`, and
      2) calling `plot_and_save` with our (x, y_pred) points as proposals.
    """
    xs = xs_cpu.view(-1).numpy()
    ys_pred = ys_pred_cpu.view(-1).numpy()

    # Match the plotting region used in training for timesTwo.
    scale = 10.0
    x_min, x_max = -scale, scale
    y_min, y_max = -2 * scale, 2 * scale
    resolution = 400

    with torch.no_grad():
        XX, YY, energy = compute_energy_grid(
            model,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            device=device,
        )

    # Use the "proposal_points" overlay machinery from `plot_timesTwo_energy.py`.
    proposal_points = {
        "xs": xs,
        "ys": ys_pred,
        # All proposals are from a single "iteration" (index 0) here.
        "iters": np.zeros_like(xs, dtype=np.float32),
    }

    title = "timesTwo predictions (ICL overlay)"
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


def check_argmin_gradients(model, device, x_values,
                           x_min=-10.0, x_max=10.0,
                           y_min=-20.0, y_max=20.0,
                           resolution=400):
    """
    For each x in x_values:
      1) Find the discrete argmin_y E(x, y) on the same grid used in the
         energy plots (via `compute_energy_grid`).
      2) Compute the *continuous* gradient of E(x, y) at that (x*, y*).

    This shows whether the red argmin curve corresponds to a true
    stationary point (grad ≈ 0) or not.
    """
    with torch.no_grad():
        XX, YY, energy = compute_energy_grid(
            model,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            device=device,
        )

    xs_axis = XX[0, :]
    ys_axis = YY[:, 0]

    print("\nGradient at grid argmin points:")
    print(" target_x | grid_x | grid_y |  dE/dy  |  dE/dx ")

    for x0 in x_values:
        # Nearest x on the plotting grid
        ix = int(np.argmin(np.abs(xs_axis - x0)))
        # Column-wise argmin over y for this x index
        col = energy[:, ix]
        iy = int(col.argmin())

        x_c = float(xs_axis[ix])
        y_c = float(ys_axis[iy])

        pair = torch.tensor([[y_c, x_c]],
                            dtype=torch.float32,
                            device=device,
                            requires_grad=True)
        e_val = model(pair).sum()
        grad, = torch.autograd.grad(e_val, pair, retain_graph=False, create_graph=False)
        dy = float(grad[0, 0].item())
        dx = float(grad[0, 1].item())

        print(f"{x0:8.3f} | {x_c:6.3f} | {y_c:6.3f} | {dy:7.4f} | {dx:7.4f}")


if __name__ == "__main__":
    # Edit this list by hand to try different inputs.
    #
    # Example:
    #   test_inputs = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    # test_inputs = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,4,5,6,7,8,9,10]
    test_inputs = np.linspace(-10, 10, 100)
    test_inputs = test_inputs.tolist()

    model, FLAGS, dataset, device = load_model_and_dataset(CKPT_PATH)

    xs_cpu, ys_pred_cpu = run_inference(
        model=model,
        FLAGS=FLAGS,
        dataset=dataset,
        device=device,
        inputs=test_inputs,
    )

    # Print numeric results.
    print("Input x  |  Predicted y  |  2x (ground truth)")
    for x, y_pred in zip(xs_cpu.view(-1).tolist(), ys_pred_cpu.view(-1).tolist()):
        print(f"{x:8.3f}  |  {y_pred:12.3f}  |  {2.0 * x:12.3f}")

    # Only plot if this is effectively 1D -> 1D (timesTwo-style) data.
    if dataset.inp_dim == 1 and dataset.out_dim == 1:
        plot_predictions_on_energy(model, device, xs_cpu, ys_pred_cpu, PLOT_PATH)
        print(f"Saved prediction + energy plot to: {PLOT_PATH}")
        # Also print gradients at the discrete argmin points for the same xs.
        check_argmin_gradients(model, device, xs_cpu.view(-1).tolist())
    else:
        print("Skipping plot because data is not 1D -> 1D.")



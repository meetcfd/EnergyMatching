import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, FlowMatcher, RectifiedFlow, ConditionalFlowMatcher
from torchcfm.utils import sample_8gaussians, sample_moons
from typing import Union

def flow_weight_schedule(t: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Optional flow weighting used during phase 2.

    We decay the flow loss linearly when ``t`` exceeds ``cutoff``:

    .. math::

       w(t) = \begin{cases}
           1, & t < \tau^*, \\
           1 - \frac{t-\tau^*}{1-\tau^*}, & \tau^* \le t < 1, \\
           0, & t \ge 1.
       \end{cases}

    This acts as an optional regularization and is disabled during phase 1.
    """

    w = torch.ones_like(t)
    decay_region = (t >= cutoff) & (t < 1.0)
    w[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
    w[t >= 1.0] = 0.0
    return w


def temperature(
    t: torch.Tensor,
    tau_star: float,
    epsilon_max: float
) -> torch.Tensor:
    """
    Piecewise definition of eps(t):
      - eps(t) = 0,                    for t < tau_star
      - eps(t) = linear ramp up to epsilon_max,  for tau_star <= t < 1
      - eps(t) = epsilon_max,          for t >= 1
    """
    if t.dim() == 2 and t.size(1) == 1:
        t = t.squeeze(-1)
    eps = torch.zeros_like(t)

    # region where we ramp from 0 up to epsilon_max
    mask_mid = (t >= tau_star) & (t < 1.0)
    scale = 1.0 - tau_star  # length of the ramp
    eps[mask_mid] = epsilon_max * (t[mask_mid] - tau_star) / scale

    # region where we remain at epsilon_max
    eps[t >= 1.0] = epsilon_max

    return eps


def velocity_training(model: nn.Module, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Compute velocity for training the flow: -∇V
    """
    x = x.detach().requires_grad_(True)
    V = model(t, x, y=kwargs.get('y', None))
    gradV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    return -gradV


def velocity_inference(model: nn.Module, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Compute velocity for inference (sampling): -∇V
    """
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        V = model(t, x, y=kwargs.get('y', None))
        gradV = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
    return -gradV


def gibbs_sampler(
    model: nn.Module,
    x_init: torch.Tensor,
    t_start: torch.Tensor,
    *,
    steps: int = 10,
    dt: float = 0.01,
    tau_star: float,
    epsilon_max: float,
    **kwargs) -> torch.Tensor:
    """
    Langevin-like sampler from t_start up to t=1.0, then at t >= 1.0 for the EBM portion.
    """
    x = x_init
    device = x_init.device

    for step in range(steps):
        # Current time from t_start up to 1.0 in discrete steps
        #   e.g. linear interpolation: t_current = t_start + (1 - t_start) * alpha
        alpha = (step + 1) / steps
        t_current = t_start + (1 - t_start) * alpha

        x.requires_grad_(True)
        V = model(torch.tensor(1.0, device=device), x, y=kwargs.get('y', None))
        g = torch.autograd.grad(V.sum(), x, create_graph=False)[0]

        eps = temperature(t_current, tau_star=tau_star, epsilon_max=epsilon_max)
        noise_scale = torch.sqrt(2.0 * eps * dt).unsqueeze(-1)
        noise = noise_scale * torch.randn_like(x)

        x = (x - g * dt + noise).detach()

    return x


def simulate_piecewise_length(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    dt: float = 0.01,
    max_length: float = 4.0,
    tau_star: float,
    epsilon_max: float,
    **kwargs
):
    """
    Incrementally simulate the system until the total path length exceeds max_length.
    For t < tau_star, no noise; for t >= tau_star, noise is added following
    a piecewise linear ramp up to epsilon_max, then constant afterwards.
    """
    x = x0
    traj = [x0.cpu().numpy()]
    times = [0.0]
    t_now = 0.0
    cum_length = 0.0
    device = x0.device

    while cum_length < max_length:
        t_tensor = torch.tensor([t_now], dtype=x0.dtype, device=device)
        g = velocity_inference(model, x, t_tensor, y=kwargs.get('y', None))
        eps_now = temperature(t_tensor, tau_star=tau_star, epsilon_max=epsilon_max).item()

        if t_now < tau_star:
            # No noise until tau_star
            dx = g * dt
        else:
            # After tau_star, use eps(t)
            noise = torch.sqrt(torch.tensor(2.0 * eps_now * dt, device=device)) * torch.randn_like(x)
            dx = g * dt + noise

        x = (x + dx).detach()
        step_length = torch.norm(dx).item()
        cum_length += step_length
        t_now += dt

        traj.append(x.cpu().numpy())
        times.append(t_now)

    return np.array(traj), np.array(times)


def plot_trajectories_custom(traj: np.ndarray) -> None:
    """
    Plots initial positions, trajectories, and final positions for a sample of the batch.
    """
    n = 2000
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    # Initial positions (black squares)
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=3, alpha=0.8, c='black', marker='s')
    # The entire trajectory
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.1, c='olive')
    # Final positions (blue stars)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1.0, c='blue', marker='*')
    # A few highlighted paths
    for i in range(10):
        plt.plot(traj[:, i, 0], traj[:, i, 1], c='red', linewidth=1.2, alpha=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def load_model_from_checkpoint(
    model_class,
    ckpt_path: str,
    device: torch.device
) -> nn.Module:
    """
    Loads a pre-trained model from a checkpoint file.

    Args:
        model_class: The class of the model to instantiate.
        ckpt_path: Path to the checkpoint file (.pth).
        device: The device to load the model onto.

    Returns:
        The loaded model.
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    # Instantiate the model with the same architecture used during training
    model = model_class(dim=2, w=256).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model

def get_batch(FM : Union[FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher], x0 : torch.Tensor, x1 : torch.Tensor, return_noise=False):
        
    if return_noise:
        t, xt, ut, noise = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
    if return_noise:
        return t[..., None], xt, ut, noise
    return t[..., None], xt, ut
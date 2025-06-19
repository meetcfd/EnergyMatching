# File: utils_train.py

import os
import time
import copy
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from absl import flags, logging
import pandas as pd

from datetime import datetime
from utils_proteins import Encoder

FLAGS = flags.FLAGS

# Set up CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def sde_epsilon(t: float, at_data_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of shape (batch_size,) with epsilon values for each sample
    at the given time t. If at_data_mask[i] == True, that sample uses epsilon_max.
    Otherwise, it follows a piecewise schedule:
      - 0 for t < FLAGS.time_cutoff
      - linearly from 0..epsilon_max as t goes from FLAGS.time_cutoff..1.0
      - constant epsilon_max for t >= 1.0
    """
    eps_max = FLAGS.epsilon_max
    cutoff = FLAGS.time_cutoff

    # Prepare output (same device/dtype as mask)
    e_val = torch.zeros_like(at_data_mask, dtype=torch.float32)
    e_val = e_val.to(at_data_mask.device)

    # For samples that are not "at data"
    not_data_mask = ~at_data_mask

    if t < cutoff:
        # they get 0 if t < cutoff
        e_val[not_data_mask] = 0.0
    elif t < 1.0:
        # linear ramp between cutoff and 1.0
        frac = (t - cutoff) / (1.0 - cutoff)
        e_val[not_data_mask] = frac * eps_max
    else:
        # constant epsilon_max if t >= 1.0
        e_val[not_data_mask] = eps_max

    # For samples that are "at data"
    e_val[at_data_mask] = eps_max

    return e_val

def gibbs_sampling_time_sweep(
    x_init: torch.Tensor,
    model,
    at_data_mask: torch.Tensor,
    n_steps: int = 150,  # e.g. 150 steps
    dt: float = 0.01,     # each step is 0.01 => total time 1.5
    clamp: bool=True # clamp when working in one-hot encoded space
):
    """
    Perform MALA sampling from t=0 to t=(n_steps*dt), with a time-dependent
    "temperature" epsilon(t). If at_data_mask[i] == True for a given sample i,
    then that sample always uses epsilon_max.

    Update rule (MALA):
      x_{k+1} = x_k - dt * ∂V/∂x + sqrt(2 * dt * epsilon(t)) * Normal(0, I).

    At the end, clamp once to the range [-1, 1].
    """
    samples = x_init.clone().detach().to(device)
    at_data_mask = at_data_mask.to(device=device, dtype=torch.bool)

    batch_size = samples.shape[0]

    for i in range(n_steps):
        t_val = i * dt  # from 0.0 up to (n_steps-1)*dt

        # Compute per-sample epsilon
        e_val = sde_epsilon(t_val, at_data_mask)
        noise_std = torch.sqrt(2.0 * dt * e_val)  # shape: (batch_size,)

        # Compute gradient of potential
        samples.requires_grad_(True)

        # Build a time tensor for potential
        t_tensor = torch.full(
            (batch_size,),
            t_val,
            device=samples.device,
            dtype=samples.dtype
        )

        V = model.potential(samples, t_tensor)
        grad_V = torch.autograd.grad(
            V,
            samples,
            grad_outputs=torch.ones_like(V),
            create_graph=False
        )[0]

        # MALA update
        with torch.no_grad():
            noise = torch.randn_like(samples)
            # Broadcast noise_std to match samples shape
            noise = noise * noise_std.view(-1, *([1]*(samples.ndim-1)))
            samples = samples - dt * grad_V + noise

    # Final clamp
    if clamp:
        samples = samples.clamp(-1.0, 1.0)
    return samples.detach()


def create_timestamped_dir(base_output_dir, model_name, append=""):
    """
    Creates a directory named like:
       base_output_dir / [model_name]_YYYYMMDD_HH
    If that directory exists, it appends _verX.
    Returns the final directory path.
    """
    ts = time.strftime('%Y%m%d_%H')  # e.g. 20250214_15
    base_name = f"{model_name}_{ts}"
    path_candidate = os.path.join(base_output_dir, base_name)

    ver_idx = 1
    while os.path.exists(path_candidate):
        path_candidate = os.path.join(
            base_output_dir, f"{base_name}_ver{ver_idx}{append}"
        )
        ver_idx += 1

    os.makedirs(path_candidate)
    return path_candidate


##############################################################################
# Time-dependent noise schedule for the SDE (plotting).
##############################################################################
def plot_epsilon(t):
    """
    A piecewise function for epsilon(t) in the *plotting* SDE:
      - 0 for t < FLAGS.time_cutoff
      - linearly from 0..epsilon_max as t goes from FLAGS.time_cutoff..1.0
      - constant epsilon_max for t >= 1.0
    """
    eps_max = FLAGS.epsilon_max
    cutoff = FLAGS.time_cutoff

    if t < cutoff:
        return 0.0
    elif t < 1.0:
        frac = (t - cutoff) / (1.0 - cutoff)  # goes from 0..1
        return frac * eps_max
    else:
        return eps_max


def sde_euler_maruyama(model, x0, t0, t1, dt=0.01, steps_to_save=None, clamp=True):
    """
    Euler–Maruyama integration from t = t0 to t1 with step dt.
    This version does NOT do an extra step if (t1 - t0) is an integer multiple of dt.
    We clamp once at the very end.
    """
    model.eval()
    times = torch.arange(t0, t1+1e-6, dt, device=device)

    x = x0.clone().to(device)
    trajectory = []

    for step_idx, t_val in enumerate(times):
        # Optionally store a copy BEFORE the update
        if steps_to_save is None or (step_idx in steps_to_save):
            trajectory.append(x.clone().detach())

        with torch.no_grad():
            # 1) Evaluate drift v(t, x)
            v = model(t_val.unsqueeze(0), x)

            # 2) Time-dependent noise scale e(t)
            e = plot_epsilon(float(t_val))
            e_tensor = torch.tensor(e, device=x.device, dtype=x.dtype)
            dt_tensor = torch.tensor(dt, device=x.device, dtype=x.dtype)
            
            # 3) Euler–Maruyama step
            noise = torch.randn_like(x)
            sigma = torch.sqrt(2.0 * e_tensor * dt_tensor)
            x = x + v * dt_tensor + sigma * noise

    # After the final step, clamp once at the very end
    if clamp:
        x = x.clamp(-1, 1)

    # Append final state
    if steps_to_save is None:
        trajectory.append(x.clone().detach())
    else:
        last_step_idx = len(times)  # "one past" the last loop index
        if last_step_idx in steps_to_save:
            trajectory[-1] = x.clone().detach()
        else:
            trajectory.append(x.clone().detach())

    # Return shape: (num_snapshots, batch, channels, height, width)
    return torch.stack(trajectory, dim=0)



def generate_samples(model, savedir, step, seq_len, net_="normal", real_data=None, vae=None):
    """
    Clones the EBM model and generates:
      1) Single-step sample (0 -> 1) with SDE (Euler–Maruyama).
      2) Time-evolution plot from t=0..3 using the SDE.
      3) Gibbs diagnostic plot (if real_data is provided).
    """
    model_clone = copy.deepcopy(model).to(device)
    model_clone.load_state_dict(model.state_dict())
    model_clone.eval()
    generate_samples_sde(model_clone, savedir, step, seq_len, net_=net_, vae=vae)


##############################################################################
# (1) Single-step sample from t=0 to t=1 with SDE
##############################################################################
def generate_samples_sde(model, savedir, step, seq_len, net_="normal", vae=None):
    """
    We'll integrate from t=0..1 with dt=0.01, then save the final images.
    """
    model.eval()
    with torch.no_grad():
        if vae is not None:
            clamp = False 
            init = torch.randn(64, 1, seq_len, device=device).permute(0,2,1)
        else:
            clamp = True
            init = torch.randn(64, seq_len, 20, device=device) # torch.randn(64, 3, 32, 32, device=device)
        traj = sde_euler_maruyama(model, init, t0=0.0, t1=1.0, dt=0.01, clamp=clamp)
        final = traj[-1]

    outpath = os.path.join(savedir, f"{net_}_generated_FM_proteins_step_{step}.csv")
    if vae is not None:
        with torch.no_grad():
            logits = vae.decode(final.squeeze())
        seqs = Encoder().decode(logits.argmax(-1))
    else:
        seqs = Encoder().decode(torch.argmax(final,-1))
    pd.Series(seqs).to_csv(outpath, index=False, header=False)
    model.train()


##############################################################################
# Helpers for training
##############################################################################
def flow_weight(t, cutoff=0.8):
    """
    Flow weighting function:
    - w_flow = 1 for t < cutoff
    - linearly from 1 down to 0 as t goes from cutoff..1
    - 0 for t >= 1
    """
    w = torch.ones_like(t)
    decay_region = (t >= cutoff) & (t < 1.0)
    w[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
    w[t >= 1.0] = 0.0
    return w


def cd_weight(t, cutoff=0.8):
    """
    Contrastive Divergence weighting function:
    - w_cd = 0 for t < cutoff
    - linearly from 0..1 as t goes from cutoff..1
    - 1 for t > 1.0
    """
    w = torch.zeros_like(t)
    region = (t >= cutoff) & (t <= 1.0)
    w[region] = (t[region] - cutoff) / (1.0 - cutoff)
    w[t > 1.0] = 1.0
    return w


def warmup_lr(step):
    """
    Simple linear warmup schedule for LR.
    """
    return min(step, FLAGS.warmup) / FLAGS.warmup


def ema(source, target, decay):
    """
    Exponential Moving Average update.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

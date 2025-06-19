import os
import random
import torch
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

from pathlib import Path
import sys
# Add repo root to path for plot_epsilon
sys.path.append(str(Path(__file__).resolve().parents[1].parent))

from absl import app, flags
import config  # assumes flags for dropout and output_scale are defined here
config.define_flags()
FLAGS = flags.FLAGS

from utils_proteins import Encoder, check_duplicates, plot_epsilon
from model_proteins import Unet1DModelWrapper, VAE
from oracle import eval, BaseCNN

# Device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Ensure working directory is script's location
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)

# Reproducibility
def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample(zeta, sigma_W, t1, n_samples, scenario, task, modality, ckpt_name):
    """
    Perform posterior sampling with classifier guidance and repulsion term.
    Returns arrays of fitness, diversity, and novelty across fixed seeds.
    """
    # Set fixed seed for consistency
    seed_all(42)
    # Five random seeds to average over
    seeds = np.random.randint(0, 1000, size=5)
    fitness_all = np.zeros(len(seeds))
    diversity_all = np.zeros(len(seeds))
    novelty_all = np.zeros(len(seeds))

    # Common parameters
    filter_top_k = True
    top_k = 128
    seq_len = 28
    latent_dim = 16
    y_gt = torch.ones((n_samples, 1, 1), dtype=torch.float32, device=device)

    # Ground truth bounds
    y_min, y_max = 0.0, 19.5365

    # VAE 
    encoder_path = Path(__file__).resolve().parents[0] / 'vae/' / f"vae_{scenario}_{task}.pt"
    vae_model = VAE(input_dim=seq_len,latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load(encoder_path, map_location=device)["state_dict"], strict=True)
    vae_model.eval()

    # Load EBM model
    model = Unet1DModelWrapper(
        dim=28, 
        channels=1,
        dim_mults=(1, 2),
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
    ).to(device)
    model.eval()

    ckpt_file = hf_hub_download(
        repo_id=ckpt_name,  
        filename=f"aav_{task}_main_training.pt",  
    )
    ckpt = torch.load(ckpt_file, map_location=device)

    # Check if the checkpoint is a dictionary
    if isinstance(ckpt, dict) and 'ema_model' in ckpt:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model.load_state_dict(ckpt)

    # Load predictor for classifier guidance
    pred = BaseCNN().to(device)
    pred_ckpt = Path(__file__).resolve().parents[0] / 'predictor' / f"{modality}" / f'predictor_{scenario}_{task}.ckpt'
    pred_state = torch.load(pred_ckpt, map_location=device)
    pred.load_state_dict({k.replace('predictor.', ''): v for k, v in pred_state['state_dict'].items()})
    pred.eval()

    # Time discretization
    dt = 0.01
    t0 = 0.0
    times = torch.arange(t0, t1 + 1e-6, dt, device=device)

    # Sampling loop over seeds
    for i, s in enumerate(seeds):
        np.random.seed(int(s))
        x = torch.randn(n_samples, 1, latent_dim, device=device).permute(0,2,1)

        for t_val in times:
            e = plot_epsilon(float(t_val))
            e_t = torch.tensor(e, device=device)
            dt_t = torch.tensor(dt, device=device)

            x.requires_grad_(True)
            v = model.potential(x, t_val.unsqueeze(0)).sum()
            y_pred = (pred.forward_soft(vae_model.decode(x.squeeze())) - y_min) / (y_max - y_min)
            likelihood = 0.5 * ((y_pred - y_gt.squeeze()) ** 2)
            cg = (e_t / zeta**2) * likelihood.sum()

            # Repulsion among samples
            if sigma_W is not None:
                x_flat = x.view(n_samples, -1)
                diffs = ((x_flat.unsqueeze(1) - x_flat.unsqueeze(0))**2).sum(-1)
                mask = ~torch.eye(n_samples, dtype=torch.bool, device=device)
                W = 0.5 * diffs[mask].sum() * (e_t / sigma_W**2)
            else:
                W = 0.0

            u = v + cg - W
            grad_x = torch.autograd.grad(u, x)[0]

            noise = torch.randn_like(x)
            sigma = torch.sqrt(2.0 * e_t * dt_t)
            with torch.no_grad():
                x = (x - dt_t * grad_x + sigma * noise).clamp(-1,1)

        # Decode sequences and remove duplicates
        x_dec = vae_model.decode(x.squeeze())
        seqs = Encoder().decode(torch.argmax(x_dec, -1))
        _, seqs = check_duplicates(seqs)

        # Optional top-k filtering
        if filter_top_k:
            fitness_preds = []
            with torch.no_grad():
                for seq in seqs:
                    f = pred.forward(Encoder().encode(seq).to(device)[None, ...]).item()
                    fitness_preds.append(f)
            fitness_tensor = torch.tensor(fitness_preds)
            k = min(top_k, len(seqs))
            _, idxs = torch.topk(fitness_tensor, k)
            seqs = [seqs[idx] for idx in idxs.tolist()]

        # Save for evaluation (required by eval)
        outpath = os.path.join(cwd, 'results', 'samples.csv')
        pd.Series(seqs).to_csv(outpath, index=False, header=False)

        # Evaluate generated samples
        f_med, div, nov_med = eval(
            scenario=scenario,
            task=task,
            baselines_samples_dir=outpath
        )
        fitness_all[i] = f_med
        diversity_all[i] = div
        novelty_all[i] = nov_med

    return fitness_all, diversity_all, novelty_all


def main(argv):
    # Sampling settings (same as original)
    sigma_W = None 
    n_samples = 512
    scenario = "aav"
    task = "medium"

    # Default parameters from the paper 
    if task == "hard":
        zeta = 0.009 
        t1 = 1.3 
        modality = "smoothed" 

    elif task == "medium": 
        zeta = 0.01 
        t1 = 1.7 
        modality = "unsmoothed" 
        
    ckpt_name = "m1balcerak/energy_matching"
    fitness_all, diversity_all, novelty_all = sample(
        zeta, sigma_W, t1, n_samples,
        scenario, task, modality, ckpt_name
    )

    # Print averaged results
    print('\nFitness %.2f (%.2f)' %(fitness_all.mean(),fitness_all.std()))
    print('Diversity %.2f (%.2f)' %(diversity_all.mean(),diversity_all.std()))
    print('Novelty %.2f (%.2f)' %(novelty_all.mean(),novelty_all.std()))

if __name__ == "__main__":
    app.run(main)

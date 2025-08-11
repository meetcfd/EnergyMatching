import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.mlp import EM_MLP_Wrapper as MLP
from physics_flow_matching.unet.mlp import ACTS
from torch.utils.data import DataLoader
from physics_flow_matching.multi_fidelity.synthetic.dataset import  flow_guidance_dists_em #flow_guidance_dists_em_class_cond
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from train import train, SampleBuffer #train_class_cond

def create_dir(path, config):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        assert config.train.restart != False, "Are you restarting?"
        print(f"Directory '{path}' already exists.")

def main(config_path):

    config = OmegaConf.load(config_path)
    
    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)
    
    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)
    
    dev = th.device(config.device)
    
    writer = SummaryWriter(log_dir=logpath)
    
    dataset = flow_guidance_dists_em(dist_name1=config.dataset.dist_name1,
                                  dist_name2=config.dataset.dist_name2, n=config.dataset.n,
                                  seed=config.dataset.seed, is_phase2=config.dataset.is_phase2, normalize=config.dataset.normalize,
                                  flip=config.dataset.flip)
    
    # dataset = flow_guidance_dists_em_class_cond(dist_name1=config.dataset.dist_name1,
    #                               dist_name2=config.dataset.dist_name2, n=config.dataset.n,
    #                               seed=config.dataset.seed, is_phase2=config.dataset.is_phase2, normalize=config.dataset.normalize,
    #                               flip=config.dataset.flip, class_cond=config.dataset.class_cond)
    
    train_dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True)
        
    model = MLP(input_dim=config.mlp.input_dim,
                hidden_dims=config.mlp.hidden_dims,
                output_dim=config.mlp.output_dim,
                act1=ACTS[config.mlp.act1] if hasattr(config.mlp, 'act1') else ACTS['relu'],
                act2=ACTS[config.mlp.act2] if hasattr(config.mlp, 'act2') else None
                )

    model.to(dev)
    
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=config.FM.sigma)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    buffer_sampler = SampleBuffer()
    
    # train_class_cond(model,
    #       writer=writer,
    #       save_dir=savepath,
    #       dataloader=train_dataloader,
    #       FM=FM,
    #       optimizer=optim,
    #       device=dev,
    #       batch_size=config.dataloader.batch_size,
    #       epochs_phase1=config.train.epochs_phase1,
    #       epochs_phase2=config.train.epochs_phase2,
    #       flow_weight=config.train.flow_weight,
    #       ebm_weight=config.train.ebm_weight,
    #       use_flow_weighting=config.train.use_flow_weighting,
    #       restart=config.train.restart,
    #       restart_epoch=config.train.restart_epoch,
    #       tau_star=config.EBM.tau_star,
    #       epsilon_max=config.EBM.epsilon_max,
    #       steps =config.EBM.steps,
    #       dt=config.EBM.dt,
    #       )
    
    train(model,
          writer=writer,
          save_dir=savepath,
          dataloader=train_dataloader,
          buffer_sampler=buffer_sampler,
          FM=FM,
          optimizer=optim,
          device=dev,
          batch_size=config.dataloader.batch_size,
          epochs_phase1=config.train.epochs_phase1,
          epochs_phase2=config.train.epochs_phase2,
          flow_weight=config.train.flow_weight,
          ebm_weight=config.train.ebm_weight,
          norm_weight=config.train.norm_weight,
          use_flow_weighting=config.train.use_flow_weighting,
          restart=config.train.restart,
          restart_epoch=config.train.restart_epoch,
          tau_star=config.EBM.tau_star,
          epsilon_max=config.EBM.epsilon_max,
          steps =config.EBM.steps,
          dt=config.EBM.dt,
          )

if __name__ == '__main__':
    main(sys.argv[1])
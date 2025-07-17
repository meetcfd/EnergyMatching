import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from experiments.toy2d.utils_2D_new import flow_weight_schedule, gibbs_sampler, velocity_training, get_batch
from physics_flow_matching.multi_fidelity.synthetic import dataset

def restart_func(restart_epoch, path, model, optimizer, sched=None):
    assert restart_epoch != None, "restart epoch not initialized!"
    print(f"Loading state from checkpoint epoch : {restart_epoch}")
    state_dict = torch.load(f'{path}/checkpoint_{restart_epoch}.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    start_epoch = restart_epoch + 1
    
    if 'sched_state_dict' in state_dict.keys() and state_dict['sched_state_dict'] is not None:
        sched.load_state_dict(state_dict['sched_state_dict']) 
        
    return start_epoch, model, optimizer, sched

def train(
    model : nn.Module,
    *,
    dataloader : DataLoader,
    optimizer : optim.Optimizer,
    FM : ExactOptimalTransportConditionalFlowMatcher,
    device: torch.device,
    batch_size: int,
    epochs_phase1: int,
    epochs_phase2: int,
    flow_weight: float,
    ebm_weight: float,
    save_dir: str,
    tau_star: float,
    epsilon_max: float,
    steps: int=200,
    dt: float=0.01,
    use_flow_weighting: bool = False,
    writer: SummaryWriter,
    restart: bool = False,
    restart_epoch: int = None,
    sched=None
) -> nn.Module:
    """
    Train the model with both OT Flow and EBM terms. Optionally
    apply time-dependent weighting to the flow loss during phase 2
    using ``use_flow_weighting``.
    """
    if restart:
        start_epoch, model, optimizer, sched = restart_func(restart_epoch, save_dir, model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
    
    # Determine update intervals for progress messages (every 20%)
    interval_p1 = max(1, epochs_phase1 // 5)
    interval_p2 = max(1, epochs_phase2 // 5)

    # --------------------------------
    # Phase 1: OT Flow matching
    # --------------------------------
    print("\n--- Phase 1: OT Flow Training Begins ---")
    dataloader.dataset.is_phase2 = False
    for i in range(start_epoch, epochs_phase1):
        total_loss = 0.0
        iteration = 0
        for x0, x1 in dataloader:
            optimizer.zero_grad()
            x0, x1 = x0.to(device), x1.to(device)
            t_samp, x_t, u_t = get_batch(FM, x0, x1)
            v_pred = velocity_training(model, x_t, t_samp)
            loss_flow = (v_pred - u_t).pow(2).mean()
            loss_flow.backward()
            optimizer.step()
            total_loss += loss_flow.item()
            iteration += 1

        writer.add_scalar('Phase1/FlowLoss', total_loss/iteration, i)
        
        if (i + 1) % interval_p1 == 0 or i == epochs_phase1 - 1:
            pct = int((i + 1) / epochs_phase1 * 100)
            print(f"Phase 1: {pct}% done (Loss: {loss_flow.item():.4f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sched_state_dict': sched.state_dict() if sched else None,
            }, f'{save_dir}/checkpoint_{i + 1}.pth')
    # --------------------------------
    # Phase 2: EBM + Flow
    # --------------------------------
    print("\n--- Phase 2: EBM + OT Flow Training Begins ---")
    dataloader.dataset.is_phase2 = True
    phase_2_start_epoch = epochs_phase1 if start_epoch < epochs_phase1 else start_epoch
    for i in range(phase_2_start_epoch, epochs_phase1+epochs_phase2):
        total_total_loss = 0.0
        total_flow_loss = 0.0
        total_ebm_loss = 0.0
        iteration = 0
        for x0, x_prior_init, x1, x_data, x_data_init in dataloader:
            optimizer.zero_grad()
            x0, x_prior_init, x1, x_data, x_data_init =  x0.to(device), x_prior_init[:batch_size//2].to(device), x1.to(device), x_data.to(device), x_data_init[:batch_size//2].to(device)
          
            # Flow portion
            t_flow, x_t_flow, u_t_flow = get_batch(FM, x0, x1)
            v_pred_flow = velocity_training(model, x_t_flow, t_flow)

            flow_mse = (v_pred_flow - u_t_flow).pow(2).mean(dim=1)
            if use_flow_weighting:
                w_flow = flow_weight_schedule(t_flow, cutoff=tau_star)
                loss_flow = torch.mean(w_flow * flow_mse)
            else:
                loss_flow = flow_mse.mean()

            # EBM portion
            # Evaluate energy at data
            Epos = model(torch.tensor(1.0, device=device), x_data).mean()

            # Negative samples are generated via gibbs_sampler
            half_bs = batch_size // 2
            x_init_neg = torch.cat([x_data_init, x_prior_init], dim=0)
            t_start = torch.cat(
                [torch.ones(half_bs, device=device), torch.zeros(half_bs, device=device)],
                dim=0
            )

            x_neg = gibbs_sampler(
                model,
                x_init_neg,
                t_start,
                steps=steps,
                dt=dt,
                tau_star=tau_star,
                epsilon_max=epsilon_max
            )
            Eneg = model(torch.tensor(1.0, device=device), x_neg).mean()

            loss_ebm = Epos - Eneg
            loss = flow_weight * loss_flow + ebm_weight * loss_ebm
            loss.backward()
            optimizer.step()
            total_total_loss += loss.item()
            total_flow_loss += loss_flow.item()
            total_ebm_loss += loss_ebm.item()
            iteration += 1
            
        writer.add_scalar('Phase2/Total_loss', total_total_loss/iteration, i)
        writer.add_scalar('Phase2/Flow_loss', total_flow_loss/iteration, i)
        writer.add_scalar('Phase2/EBM_loss', total_ebm_loss/iteration, i)
        
        if (i + 1) % interval_p2 == 0 or i == epochs_phase2 - 1:
            pct = int((i + 1) / epochs_phase2 * 100)
            print(f"Phase 2: {pct}% done (Total Loss: {loss.item():.4f}, Flow: {loss_flow.item():.4f}, EBM: {loss_ebm.item():.4f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sched_state_dict': sched.state_dict() if sched else None,
            }, f'{save_dir}/checkpoint_{i + 1}.pth')

    print(f"\nTraining complete!")
    return model

def train_class_cond(
    model : nn.Module,
    *,
    dataloader : DataLoader,
    optimizer : optim.Optimizer,
    FM : ExactOptimalTransportConditionalFlowMatcher,
    device: torch.device,
    batch_size: int,
    epochs_phase1: int,
    epochs_phase2: int,
    flow_weight: float,
    ebm_weight: float,
    save_dir: str,
    tau_star: float,
    epsilon_max: float,
    steps: int=200,
    dt: float=0.01,
    use_flow_weighting: bool = False,
    writer: SummaryWriter,
    restart: bool = False,
    restart_epoch: int = None,
    sched=None
) -> nn.Module:
    """
    Train the model with both OT Flow and EBM terms. Optionally
    apply time-dependent weighting to the flow loss during phase 2
    using ``use_flow_weighting``.
    """
    if restart:
        start_epoch, model, optimizer, sched = restart_func(restart_epoch, save_dir, model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
    
    # Determine update intervals for progress messages (every 20%)
    interval_p1 = max(1, epochs_phase1 // 5)
    interval_p2 = max(1, epochs_phase2 // 5)

    # --------------------------------
    # Phase 1: OT Flow matching
    # --------------------------------
    print("\n--- Phase 1: OT Flow Training Begins ---")
    dataloader.dataset.is_phase2 = False
    for i in range(start_epoch, epochs_phase1):
        total_loss = 0.0
        iteration = 0
        for x0, x1, y in dataloader:
            optimizer.zero_grad()
            x0, x1 = x0.to(device), x1.to(device)
            y = y[..., None].to(device)
            t_samp, x_t, u_t = get_batch(FM, x0, x1)
            v_pred = velocity_training(model, x_t, t_samp, y=y)
            loss_flow = (v_pred - u_t).pow(2).mean()
            loss_flow.backward()
            optimizer.step()
            total_loss += loss_flow.item()
            iteration += 1

        writer.add_scalar('Phase1/FlowLoss', total_loss/iteration, i)
        
        if (i + 1) % interval_p1 == 0 or i == epochs_phase1 - 1:
            pct = int((i + 1) / epochs_phase1 * 100)
            print(f"Phase 1: {pct}% done (Loss: {loss_flow.item():.4f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sched_state_dict': sched.state_dict() if sched else None,
            }, f'{save_dir}/checkpoint_{i + 1}.pth')
            
    # --------------------------------
    # Phase 2: EBM + Flow
    # --------------------------------
    print("\n--- Phase 2: EBM + OT Flow Training Begins ---")
    dataloader.dataset.is_phase2 = True
    phase_2_start_epoch = epochs_phase1 if start_epoch < epochs_phase1 else start_epoch
    for i in range(phase_2_start_epoch, epochs_phase1+epochs_phase2):
        total_total_loss = 0.0
        total_flow_loss = 0.0
        total_ebm_loss = 0.0
        iteration = 0
        for x0, x_prior_init, x1, x_data, x_data_init, y in dataloader:
            optimizer.zero_grad()
            x0, x_prior_init, x1, x_data, x_data_init =  x0.to(device), x_prior_init[:batch_size//2].to(device), x1.to(device), x_data.to(device), x_data_init[:batch_size//2].to(device)
            y = y[..., None].to(device)
            # Flow portion
            t_flow, x_t_flow, u_t_flow = get_batch(FM, x0, x1)
            v_pred_flow = velocity_training(model, x_t_flow, t_flow, y=y)

            flow_mse = (v_pred_flow - u_t_flow).pow(2).mean(dim=1)
            if use_flow_weighting:
                w_flow = flow_weight_schedule(t_flow, cutoff=tau_star)
                loss_flow = torch.mean(w_flow * flow_mse)
            else:
                loss_flow = flow_mse.mean()

            # EBM portion
            # Evaluate energy at data
            Epos = model(torch.tensor(1.0, device=device), x_data, y=y).mean()

            # Negative samples are generated via gibbs_sampler
            half_bs = batch_size // 2
            x_init_neg = torch.cat([x_data_init, x_prior_init], dim=0)
            t_start = torch.cat(
                [torch.ones(half_bs, device=device), torch.zeros(half_bs, device=device)],
                dim=0
            )

            x_neg = gibbs_sampler(
                model,
                x_init_neg,
                t_start,
                steps=steps,
                dt=dt,
                tau_star=tau_star,
                epsilon_max=epsilon_max,
                y=y
            )
            Eneg = model(torch.tensor(1.0, device=device), x_neg, y=y).mean()

            loss_ebm = Epos - Eneg
            loss = flow_weight * loss_flow + ebm_weight * loss_ebm
            loss.backward()
            optimizer.step()
            total_total_loss += loss.item()
            total_flow_loss += loss_flow.item()
            total_ebm_loss += loss_ebm.item()
            iteration += 1
            
        writer.add_scalar('Phase2/Total_loss', total_total_loss/iteration, i)
        writer.add_scalar('Phase2/Flow_loss', total_flow_loss/iteration, i)
        writer.add_scalar('Phase2/EBM_loss', total_ebm_loss/iteration, i)
        
        if (i + 1) % interval_p2 == 0 or i == epochs_phase2 - 1:
            pct = int((i + 1) / epochs_phase2 * 100)
            print(f"Phase 2: {pct}% done (Total Loss: {loss.item():.4f}, Flow: {loss_flow.item():.4f}, EBM: {loss_ebm.item():.4f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sched_state_dict': sched.state_dict() if sched else None,
            }, f'{save_dir}/checkpoint_{i + 1}.pth')

    print(f"\nTraining complete!")
    return model
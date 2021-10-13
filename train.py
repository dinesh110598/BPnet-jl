import torch
from BPnetModel import BPnet, sample, log_prob
from tqdm import tqdm
from ising import IsingParams, H

def train_loop(model: BPnet, tot_steps: int, beta: float, pars: IsingParams,
               batch_size=100, anneal=False, init_eta=0.005, beta_anneal=0.95):
    params = model.parameters()
    opt = torch.optim.Adam(params, lr=init_eta, betas=(0.9, 0.999))
    lr_schedule = torch.optim.lr_scheduler.StepLR(opt, 250, 0.8)

    beta_conv = beta
    outer = tqdm(total=tot_steps, desc='Training steps', position=0)
    F_log = tqdm(total=tot_steps, position=1, bar_format='{desc}')
    
    for step in range(tot_steps):
        if anneal:
            beta = beta_conv * (1 - beta_anneal**(step+1))
        
        with torch.no_grad():
            x = sample(model, pars.L, batch_size)
            E = H(x, pars)
        lp = log_prob(model, x)
        with torch.no_grad():
            loss = lp + beta*E
            F = loss / beta
        loss_reinf = torch.mean((loss - loss.mean()) * lp)
        loss_reinf.backward()
        opt.step()
        lr_schedule.step()
        
        mean_F = F.mean()
        std_F = F.std()
        outer.update()
        F_log.set_description_str(f'Average F: {mean_F} \t Std F: {std_F}')
        if (step+1)%500 == 0:
            torch.save(model.state_dict(), f"Saves/Chkpts/Step_{step}.pt")
        

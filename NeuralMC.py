import torch
from BPnetModel import BPnet, log_prob
from ising import H

def neural_mc(model: BPnet, L, batch_size, beta, n=1):
    lp = torch.empty(0, dtype=torch.float64, device="cpu")
    E = torch.empty(0, dtype=torch.float64, device="cpu")
    m = torch.empty(0, dtype=torch.float64, device="cpu")
    
    for _ in range(n):
        x2 = model.sample(L, batch_size)
        lp2 = log_prob(model, x2).to("cpu")
        e2 = H(x2).to("cpu")
        m2 = torch.mean(x2, dim=(1,2,3)).to("cpu")
        lp = torch.cat([lp, lp2])
        m = torch.cat([m, m2])
        E = torch.cat([E, e2])
        
    ptr = 0
    acc = 0
    mag = torch.empty(batch_size*n, dtype=torch.float64, device="cpu")
    ener = torch.empty(batch_size*n, dtype=torch.float64, device="cpu")
    mag[0] = m[0]
    ener[0] = E[0]
    for j in torch.arange(1, n*batch_size):
        arg = beta*(E[ptr] - E[j]) + (lp[ptr] - lp[j])
        if torch.rand() < torch.exp(arg):
            ptr = j
            acc += 1
        mag[j] = m[ptr]
        ener[j] = E[ptr]
    return torch.abs(mag), ener
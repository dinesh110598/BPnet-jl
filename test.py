#%%
import numpy as np
import torch
from BPnetModel import BPnet
from train import train_loop
from ising import IsingParams, H

print(torch.cuda.is_available())
my_device = torch.device("cuda:0")
#%%
L = 32
beta = 0.5
pars = IsingParams(L)
model = BPnet(18, 16, 2).to(my_device)
# %%
scripted_model = torch.jit.script(model)
# %%
train_loop(scripted_model, 100, beta, pars)

# %%

#%%
import torch
from BPnetModel import BPnet, log_prob
from train import train_loop
from ising import H
# %%
print(torch.cuda.is_available())
my_device = torch.device("cuda:0")
# %%
L = 64
betas = [0.4, 0.4407, 0.46, 0.5]
for beta in betas:
    model = BPnet(9, 16, 2).to(my_device)
    model.load_state_dict(torch.load(f'Saves/mish_(9,16,2)/L32_b_{beta}.pt'))
    train_loop(model, 250, beta, L)
    torch.save(model.state_dict(), f'Saves/mish_(9,16,2)/L64_b_{beta}.pt')
# %%
torch.set_grad_enabled(False)
#%%
model = BPnet(9, 16, 2).to(my_device)
model.load_state_dict(torch.load(f'Saves/Trial_L32.pt'))
# %%
beta = 0.5
x = model.sample(32, 1000)
lp = log_prob(model, x)
E = H(x, pbc=False)
F = (lp/beta + E)/(32**2)
print(F.mean())
# %%

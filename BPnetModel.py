from torch import nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, p, n, mask='B', last_layer=False):
        super().__init__()
        if mask=='A':
            self.res = 0
        else:
            self.res = 1
        self.n = n
        self.p = p
        self.last_layer = last_layer
        r = self.res
        
        self.l_conv = nn.ModuleList([nn.Conv2d(1 if r == 0 else p, p, (1, n-1+r)),
                       nn.Conv2d(p, p, (1, 1))])
        self.r_conv = nn.ModuleList([nn.Conv2d(1 if r == 0 else p, p, (1, n)),
                       nn.Conv2d(p, p, (1, 1))])
        self.m_conv = nn.ModuleList([nn.Conv2d(1 if r == 0 else p, p, (1, 1), bias=False),
                       nn.Conv2d(p, p, (1, 1)),
                       nn.Conv2d(p, p, (1, 1), bias=False)])
        self.upper_pad = nn.ZeroPad2d((0, 0, 1, 0))
        self.left_pad = nn.ZeroPad2d((n - 1, 0, 0, 0))
        self.right_pad = nn.ZeroPad2d((0, n - 1, 0, 0))
        self.activation = nn.Mish()
    
    def forward(self, x_l, x_m, x_r):
        x_m2 = torch.clone(x_m)
        if not self.res:
            x_l = x_l[:, :, :, :-1]
            x_r = x_r[:, :, :-1, :]
            x_r = self.upper_pad(x_r)
        x_l = self.left_pad(x_l)
        x_r = self.right_pad(x_r)

        x_l = self.activation(self.l_conv[0](x_l))
        x_r = self.activation(self.r_conv[0](x_r))
        x_m = self.activation(self.m_conv[0](x_m))
        x_m += x_l + x_r

        x_m = self.activation(self.m_conv[1](x_m))
        if self.res:
            x_m2 = self.activation(self.m_conv[2](x_m2))
            x_m += x_m2
        if not self.last_layer:
            x_l = self.activation(self.l_conv[1](x_l))
            x_r = self.activation(self.r_conv[1](x_r))
        return x_l, x_m, x_r


class BPnet(nn.Module):
    def __init__(self, kernel_size: int, net_width: int, net_depth: int):
        super().__init__()
        self.n = kernel_size
        self.p = net_width
        self.d = net_depth
        self.blocks = nn.ModuleList([ConvBlock(net_width, kernel_size, 'A' if i==0 else 'B', 
                                    i==net_depth-1) for i in range(net_depth)])
        self.final_conv = nn.Sequential(nn.Conv2d(net_width, 1, (1, 1)),
                                        nn.Sigmoid())

    def forward(self, x):
        x_l = x
        x_r = torch.clone(x)
        x_m = torch.zeros_like(x)
        for block in self.blocks:
            x_l, x_m, x_r = block.forward(x_l, x_m , x_r)
        return self.final_conv(x_m)

    def sample(self, L: int, batch_size: int):
        """Samples a configuration from the variational distribution associated with the neural network
        """
        x = torch.zeros(batch_size, 1, L, L, device="cuda")
        r = self.d*(self.n - 1)
        for i in torch.arange(L):
            for j in torch.arange(L):
                x_l = x[:, :,torch.maximum(i-1, i*0) : i+1, 
                        torch.maximum(j-r, i*0):torch.minimum(j+r+1, torch.tensor(L))]
                x_hat = self.forward(x_l)
                i_h = torch.min(torch.tensor([i, 1], device="cuda"))
                j_h = torch.min(torch.tensor([j, r], device="cuda"))
                probs = (torch.empty((batch_size,), device="cuda").fill_(0.5)
                            if (i, j) == (0, 0) else x_hat[:, 0, i_h, j_h])
                x[:, 0, i, j] = 2*torch.bernoulli(probs) - 1
        
        #Enforce Z2 symmetry
        probs = torch.empty((batch_size, 1, 1, 1), device="cuda").fill_(0.5)
        x = x * (2*torch.bernoulli(probs) - 1)
        return x

def _log_prob(x, x_hat):
    mask = (x + 1)/2
    log_prob = (torch.log(x_hat + 1e-5)*mask + 
                torch.log(1 - x_hat + 1e-5)*(1 - mask))
    log_prob = torch.sum(log_prob, (1, 2, 3))
    return log_prob

def log_prob(model: BPnet, x: torch.Tensor):
    """Calculates logarithm of the variational probability of
    given input, x 
    """
    x_hat = model.forward(x)
    lp = _log_prob(x, x_hat)
    #Z2 symmetry
    x_inv = -x
    x_hat_inv = model.forward(x_inv)
    lp_inv = _log_prob(x_inv, x_hat_inv)
    z = torch.stack([lp, lp_inv], dim=0)
    lp = torch.logsumexp(z, dim=0)
    return lp

import torch

class IsingParams:
    def __init__(self, L=32, pbc=False, lattice="square", J=1.,
                 nn=False):
        """
        Configues various physical parameters of the Ising model under 
        study.
        """
        self.L = L
        self.pbc = pbc
        self.lattice = lattice
        self.J = J
        self.next_nearest = nn


def H(x: torch.Tensor, pbc=True, lattice="square", J=1.) -> torch.Tensor:
    """Computes batchwise energy/Hamiltonian of the input x

    Args:
        x (torch.Tensor): A 4D tensor that holds the spin configuration
        of Ising model whose energy is to be computed.
        pbs (bool): Periodic boundary condition if true else zero boundary
        lattice (str): Lattice type/shape
        J (float): Nearest neighbor coupling constant

    Returns:
        torch.Tensor: The (batchwise) 1D energy tensor
    """
    if pbc:
        x_n = torch.roll(x, 1, dims=2)
        x_n += torch.roll(x, 1, dims=3)
        if lattice=="tri":
            x_n += torch.roll(x, (1,1), dims=(2,3))
        energy = -J * x * x_n
        energy = torch.sum(energy, dim=(1,2,3))
    else:
        n_sum = torch.sum(x[:, :, 1:, :] * x[:, :, :-1, :], dim=(1,2,3))
        n_sum += torch.sum(x[:, :, :, 1:] * x[:, :, :, :-1], dim=(1,2,3))
        if lattice=="tri":
            n_sum += torch.sum(x[:, :, :-1, :-1] * x[:, :, 1:, 1:],
                               dim=(1,2,3))
        energy = -J * n_sum
    return energy
    
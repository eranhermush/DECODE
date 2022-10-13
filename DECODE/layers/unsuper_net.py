import torch

from DECODE.layers.unsuper_layer import UnsuperLayer
import torch.nn as nn

EPSILON = torch.finfo(torch.float32).eps


class UnsuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
        super(UnsuperNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList([UnsuperLayer(comp, features, l_1, l_2) for i in range(self.n_layers)])
        self.l_1 = l_1
        self.l_2 = l_2

    def forward(self, h, x):
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        h = h / (torch.clamp(h.sum(axis=1)[:, None], min=1e-12))
        return h

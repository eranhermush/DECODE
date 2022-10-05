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
        # self.softmax = nn.Softmax(1)

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        w = None
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
            numerator = h.T.matmul(x)
            if i == 0:
                delta = l.fc2.weight
            else:
                delta = w
            denominator = torch.add(h.T.matmul(h).matmul(delta), EPSILON)
            div = torch.div(numerator, denominator)
            w = torch.mul(delta, div)
        w = w.detach().clone()
        h = h / (torch.clamp(h.sum(axis=1)[:, None], min=1e-12))
        d = x - h.matmul(w)
        loss = (0.5 * torch.pow(d, 2).sum() + self.l_1 * h.sum() + 0.5 * self.l_2 * torch.pow(h, 2).sum()) / h.shape[0]
        return h, loss

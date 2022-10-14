from typing import Tuple, List

import numpy as np
import torch
from numpy import ndarray
from scipy.optimize import nnls
from torch import tensor, optim
from torch.optim import Optimizer

from DECODE.layers.unsuper_net import UnsuperNet
from DECODE.preprocessing.decode_config import DecodeConfig

EPSILON = torch.finfo(torch.float32).eps
CELL_MAPPER = {
    "B Cells": [5, 10],
    "CD8 T Cells": [5, 30],
    "CD4 T Cells": [25, 60],
    "NK Cells": [10, 30],
    "Monocytes": [5, 10],
    "Neutrophils": [40, 60],
}


def generate_dists(signature_data: ndarray, std: float, mix_size: int, cells: List[str]) -> Tuple[ndarray, ndarray]:
    """
    generates synthetic data
    uses known cell fractions from
        https://www.miltenyibiotec.com/US-en/resources/macs-handbook/human-cells-and-organs/human-cell-sources/blood-human.htmL
    """
    dist = _get_randon_scope(cells, (mix_size, signature_data.shape[0]))

    mix = dist.dot(signature_data)
    mix += np.random.normal(0, std, mix.shape)
    mix = np.maximum(mix, 0)
    return mix.T, dist


def _init_decode(ref_mat: ndarray, mix_max: ndarray, config: DecodeConfig) -> Tuple[UnsuperNet, tensor, Optimizer]:
    features, n_components = ref_mat.shape
    samples, features = mix_max.shape
    deep_nmf = UnsuperNet(config.num_layers, n_components, features, config.l1_regularization, config.l2_regularization)
    h_0_train = _tensoring(
        np.asanyarray(
            [np.random.dirichlet(np.random.randint(1, 20, size=n_components)) for i in range(samples)], dtype=float
        )
    )
    if config.use_w0:
        deep_nmf_params = list(deep_nmf.parameters())
        for w_index in range(len(deep_nmf_params)):
            w = deep_nmf_params[w_index]
            if w_index == 0:
                w.data = _tensoring(np.dot(ref_mat.T, ref_mat))
            elif w_index == 1:
                w.data = _tensoring(ref_mat.T)

        h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
        h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train]))
    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=config.lr)
    return deep_nmf, h_0_train, optimizerADAM


def _tensoring(matrix: ndarray) -> tensor:
    """
    Convert numpy array to torch tensor
    """
    return torch.from_numpy(matrix).float()


def _get_randon_scope(cells: List[str], result_dim: Tuple[int, int]) -> ndarray:
    small_mapper = lambda key: CELL_MAPPER.get(key, [0, 20])[0]
    big_mapper = lambda key: CELL_MAPPER.get(key, [0, 20])[1]
    small_vector = np.vectorize(small_mapper)(cells)
    big_vector = np.vectorize(big_mapper)(cells)

    diff = big_vector - small_vector
    random_vector = np.random.random_sample(result_dim)
    random_vector = random_vector * diff + small_vector
    dirichlet = np.asanyarray([np.random.dirichlet(random_vector[i]) for i in range(len(random_vector))])
    return dirichlet


def cost_tns(v, w, h, l_1=0, l_2=0):
    d = v - h.mm(w)
    return (0.5 * torch.pow(d, 2).sum() + l_1 * h.sum() + 0.5 * l_2 * torch.pow(h, 2).sum()) / (v.shape[0] * v.shape[1])

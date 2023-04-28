import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import nnls
from torch import nn, tensor, optim
from torch.optim import Optimizer

from DECODE.layers.unsuper_net import UnsuperNet
from DECODE.preprocessing.data_formatter import format_dataframe
from DECODE.preprocessing.decode_config import UnsupervisedLearner, DecodeConfig
from DECODE.preprocessing.gedit_preprocessing import _quantile_normalize, _normalize_zero_one, run_gedit_preprocessing
from DECODE.train.train_utils import _tensoring, generate_dists, cost_tns, _init_decode


def train_manager(config: DecodeConfig, to_train: bool = True) -> Optional[UnsupervisedLearner]:
    """
    In this function we load all the relevant matrices, run Gedit's preprocessing and start the learning process
    """
    if (not _create_output_file(config)) and to_train:
        return
    ref_panda = pd.read_csv(config.ref_path, sep="\t", index_col=0)
    ref_panda.index = ref_panda.index.str.upper()

    mix_panda_original = pd.read_csv(config.mix_path, sep="\t", index_col=0)
    mix_panda_original.index = mix_panda_original.index.str.upper()
    mix_panda_original.columns = mix_panda_original.columns.str.upper()

    dist_panda = pd.read_csv(config.dist_path, sep="\t", index_col=0)
    dist_panda.index = mix_panda_original.columns
    dist_panda.index = dist_panda.index.str.upper()

    ref_panda_gedit, mix_panda, indexes, dist_panda = _preprocess_dataframes(
        ref_panda, mix_panda_original, dist_panda, config
    )
    original_ref, dist_panda, _ = format_dataframe(ref_panda, dist_panda)
    original_ref = original_ref.loc[indexes]
    original_ref = original_ref.sort_index(axis=0)

    if to_train:
        learner = _train_decode(mix_panda.T, ref_panda_gedit, original_ref.to_numpy(), dist_panda, config)
        return learner
    else:
        deep_nmf_original, h_0_train, optimizer = _init_decode(ref_panda_gedit, mix_panda.T, config)
        learner = UnsupervisedLearner(
            config, deep_nmf_original, optimizer, mix_panda.T, dist_panda, h_0_train, ref_panda_gedit
        )
        return learner


def _train_decode(
    mix_max: ndarray,
    ref_mat: ndarray,
    original_ref: ndarray,
    dist_mix_i: DataFrame,
    config: DecodeConfig,
) -> UnsupervisedLearner:
    """
    This function runs the training logic
    1. init a new DECODE object
    2. runs supervised training (with synthetic data)
    3. runs unsupervised training (with synthetic data)

    As a cache, the function saves a pickle of the trained models
    :param mix_max: bulk expression matrix after preprocessing
    :param ref_mat: signature matrix after preprocessing
    :param original_ref: signature matrix before preprocessing
    :param dist_mix_i: our desired cells
    :param config: config object of current training
    """
    print(f'start time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    deep_nmf_original, h_0_train, optimizer = _init_decode(ref_mat, mix_max, config)
    deep_nmf_original.to(config.device)
    original_ref = original_ref[(original_ref != 0).any(axis=1)]
    supervised_path = Path(str(config.output_path) + "GENERATED.pkl")
    cells = dist_mix_i.columns.tolist()
    if not supervised_path.is_file():
        train_with_generated_data(ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config, cells)

    assert supervised_path.is_file()
    with open(str(supervised_path), "rb") as input_file:
        checkpoint = pickle.load(input_file)
        deep_nmf_original = checkpoint["deep_nmf"]

    optimizer = optim.Adam(deep_nmf_original.parameters(), lr=config.lr)
    unsupervised_path = Path(str(config.output_path) + "_GENERATED-UNsup_new_loss_algo.pkl")
    if not (unsupervised_path.is_file()):
        normalize_loss = train_with_generated_data_unsupervised(
            ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config, unsupervised_path, cells
        )
        checkpoint = {"deep_nmf": deep_nmf_original, "config": config, "normalize_loss": normalize_loss}
        if not unsupervised_path.is_file():
            with open(unsupervised_path, "wb") as f:
                pickle.dump(checkpoint, f)
    learner = UnsupervisedLearner(config, deep_nmf_original, optimizer, mix_max, dist_mix_i, h_0_train, ref_mat)
    return learner


def train_with_generated_data(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNet,
    optimizer: Optimizer,
    config: DecodeConfig,
    cells: List[str],
):
    """
    Supervised training
    - generates synthetic data
    - run Gedit preprocessing on it (as we do in the real data)
    - runs Dnmf on it
    """
    rows, genes = mix_max.shape
    supervised_train = config.supervised_train
    for train_index in range(1, supervised_train + 1):
        generated_mix, generated_dist = generate_dists(original_ref_mat.T, train_index * 0.0001, rows, cells)
        mix_max = _quantile_normalize(generated_mix, original_ref_mat)
        _, mix_frame = _normalize_zero_one(original_ref_mat, mix_max)
        generated_dist = _tensoring(generated_dist).to(config.device)
        loss_values = train_supervised_one_sample(
            ref_mat, mix_frame.T, generated_dist, deep_nmf, optimizer, config.device
        )
        if train_index == supervised_train:
            checkpoint = {"deep_nmf": deep_nmf, "config": config}
            with open(str(config.output_path) + "GENERATED.pkl", "wb") as f:
                pickle.dump(checkpoint, f)


def train_supervised_one_sample(
    ref_mat: ndarray, mix_max: ndarray, dist_mat: tensor, deep_nmf: UnsuperNet, optimizer: Optimizer, device
):
    h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
    h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(device)

    criterion = nn.MSELoss(reduction="mean")
    mix_max = _tensoring(mix_max).to(device)
    out = deep_nmf(h_0_train, mix_max)
    loss = torch.sqrt(criterion(out, dist_mat))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for w in deep_nmf.parameters():
        w.data = w.clamp(min=0, max=math.inf)
    return loss.item()


def train_with_generated_data_unsupervised(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNet,
    optimizer: Optimizer,
    config: DecodeConfig,
    pickle_path: Path,
    cells: List[str],
) -> float:
    """
    Unsupervised training
    - generates synthetic data
    - run Gedit preprocessing on it (as we do in the real data)
    - runs Dnmf on it
    - finds supervised and unsupervised loss
    - update the NNLS loss (if twe exceed the NNLS loss we break)
    - update the network
    """
    torch.autograd.set_detect_anomaly(True)
    rows, genes = mix_max.shape
    best_normalize_matrix_loss = 2
    nnls_error = 0
    nnls_criterion = nn.MSELoss(reduction="mean")
    for train_index in range(1, config.supervised_train + 1):
        generated_mix, generated_dist = generate_dists(original_ref_mat.T, train_index * 0.0001, rows, cells)
        mix_max = _quantile_normalize(generated_mix, original_ref_mat)
        _, mix_max = _normalize_zero_one(original_ref_mat, mix_max)

        generated_dist = _tensoring(generated_dist).to(config.device)
        mix_max = mix_max.T
        h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
        h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(config.device)

        mix_max = _tensoring(mix_max).to(config.device)
        out = deep_nmf(h_0_train, mix_max)

        features = mix_max.shape[1]
        w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        decode_w_matrix = torch.from_numpy(nnls_w).float()

        loss = cost_tns(mix_max, decode_w_matrix, out, config.l1_regularization, config.l2_regularization)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            loss2, normalize_matrix_loss, normalize_out_matrix = _find_loss(out, generated_dist)

        if train_index < 1000:
            nnls_error += torch.sqrt(nnls_criterion(h_0_train, generated_dist))
        if train_index == 1000:
            nnls_error = nnls_error / 1000
        if train_index > 1000 and nnls_error < normalize_matrix_loss:
            return normalize_matrix_loss
    return normalize_matrix_loss


def train_unsupervised_real_data(
    learner: UnsupervisedLearner,
) -> Tuple[tensor, tensor, UnsuperNet, tensor, int, int]:
    deep_nmf = learner.deep_nmf
    config = learner.config
    mix_max = _tensoring(learner.mix_max).to(config.device)
    optimizer = learner.optimizer
    h_0 = learner.h_0.to(config.device)

    inputs = (h_0, mix_max)
    torch.autograd.set_detect_anomaly(True)
    out = ndarray([])
    normalize_output = ndarray([])
    best_normalize_obj = out
    best_i = 0
    total_loss = None
    nnls_result = [nnls(learner.ref, mix_max[kk])[0] for kk in range(len(mix_max))]
    nnls_result = _tensoring(np.asanyarray([d / sum(d) for d in nnls_result])).to(config.device)
    for i in range(config.supervised_train):
        out = deep_nmf(*inputs)
        features = mix_max.shape[1]
        w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        decode_w_matrix = torch.from_numpy(nnls_w).float()

        total_loss = cost_tns(mix_max, decode_w_matrix, out, config.l1_regularization, config.l2_regularization)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            _, _, normalize_output = _find_loss(out, None)
    return out, normalize_output, deep_nmf, best_normalize_obj, best_i, total_loss.item()


def _find_loss(
    out: tensor,
    dist_mix_i_tensor: Optional[tensor],
) -> Tuple[tensor, float, tensor]:
    criterion = nn.MSELoss(reduction="mean")
    loss2 = -1
    if dist_mix_i_tensor is not None:
        loss2 = torch.sqrt(criterion(out, dist_mix_i_tensor))

    normalize_out = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12))
    normalize_matrix_loss = -1
    if dist_mix_i_tensor is not None:
        normalize_matrix_loss = torch.sqrt(criterion(normalize_out, dist_mix_i_tensor))

    return loss2, normalize_matrix_loss, normalize_out


def _create_output_file(config: DecodeConfig) -> bool:
    if not os.path.isdir(config.output_path.parent):
        os.makedirs(config.output_path.parent, exist_ok=True)
    if not os.path.isdir(config.unsup_output_path.parent):
        os.makedirs(config.unsup_output_path.parent, exist_ok=True)
    if not config.rewrite_exists_output and os.path.exists(config.output_path):
        return False
    return True


def _preprocess_dataframes(
    ref_panda: DataFrame, mix_panda: DataFrame, dist_panda: DataFrame, config: DecodeConfig, use_all_genes: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataFrame]:
    ref_panda, dist_panda, mix_panda = format_dataframe(ref_panda, dist_panda, mix_panda)
    ref_panda, mix_panda, indexes = run_gedit_preprocessing(mix_panda, ref_panda, config.total_sigs, use_all_genes)
    return ref_panda, mix_panda, indexes, dist_panda

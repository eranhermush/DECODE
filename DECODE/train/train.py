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
from DECODE.preprocessing.dnmf_config import UnsupervisedLearner, DnmfConfig
from DECODE.preprocessing.gedit_preprocessing import _quantile_normalize, _normalize_zero_one, run_gedit_preprocessing
from DECODE.train.train_utils import _tensoring, generate_dists, cost_tns, _init_dnmf

SUPERVISED_SPLIT = 20000


def train_manager(config: DnmfConfig, to_train: bool = True) -> Optional[UnsupervisedLearner]:
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
        learner = _train_dnmf(mix_panda.T, ref_panda_gedit, original_ref.to_numpy(), dist_panda, config)
        return learner
    else:
        deep_nmf_original, h_0_train, optimizer = _init_dnmf(ref_panda_gedit, mix_panda.T, config)
        learner = UnsupervisedLearner(
            config, deep_nmf_original, optimizer, mix_panda.T, dist_panda, h_0_train, ref_panda_gedit
        )
        return learner


def _handle_final_matrix(
    config: DnmfConfig, out: tensor, out34: tensor, deep_nmf: UnsuperNet, dist_panda: DataFrame
) -> None:
    dist_new = dist_panda.copy()
    criterion = nn.MSELoss(reduction="mean")
    loss = torch.sqrt(criterion(out, _tensoring(dist_panda.values).to(config.device)))
    loss34 = torch.sqrt(criterion(out34, _tensoring(dist_panda.values).to(config.device)))
    checkpoint = {"loss": loss, "loss34": loss34, "deep_nmf": deep_nmf, "config": config}
    print(f"final results: loss: {loss}, loss34: {loss34}")
    with open(str(config.output_path) + ".pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    dist_new[:] = out.cpu().detach().numpy()
    dist_new.to_csv(config.output_path, sep="\t")

    dist_new[:] = out34.cpu().detach().numpy()
    dist_new.to_csv(str(config.output_path) + "34.tsv", sep="\t")

    print(f'Finish time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')


def _create_output_file(config: DnmfConfig) -> bool:
    if not os.path.isdir(config.output_path.parent):
        os.makedirs(config.output_path.parent, exist_ok=True)
    if not os.path.isdir(config.unsup_output_path.parent):
        os.makedirs(config.unsup_output_path.parent, exist_ok=True)
    if not config.rewrite_exists_output and os.path.exists(config.output_path):
        return False
    return True


def _preprocess_dataframes(
    ref_panda: DataFrame, mix_panda: DataFrame, dist_panda: DataFrame, config: DnmfConfig, use_all_genes: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataFrame]:
    ref_panda, dist_panda, mix_panda = format_dataframe(ref_panda, dist_panda, mix_panda)
    ref_panda, mix_panda, indexes = run_gedit_preprocessing(mix_panda, ref_panda, config.total_sigs, use_all_genes)
    return ref_panda, mix_panda, indexes, dist_panda


def _train_dnmf(
    mix_max: ndarray,
    ref_mat: ndarray,
    original_ref: ndarray,
    dist_mix_i: DataFrame,
    config: DnmfConfig,
) -> UnsupervisedLearner:
    print(f'start time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    deep_nmf_original, h_0_train, optimizer = _init_dnmf(ref_mat, mix_max, config)
    deep_nmf_original.to(config.device)
    original_ref = original_ref[(original_ref != 0).any(axis=1)]
    tmp = config.unsupervised_train
    supervised_path = Path(str(config.output_path) + "GENERATED.pkl")
    config.unsupervised_train = tmp
    supervised_all_iterations = config.supervised_train
    cells = dist_mix_i.columns.tolist()
    if supervised_path.is_file():
        print("Supervised file exsists")
        with open(str(supervised_path), "rb") as input_file:
            checkpoint = pickle.load(input_file)
            deep_nmf_original = checkpoint["deep_nmf"]
    else:
        print("Start supervised")
        train_with_generated_data(ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config, cells)

    for supervised_index in range(SUPERVISED_SPLIT, supervised_all_iterations + 1, SUPERVISED_SPLIT):
        config.supervised_train = supervised_index
        tmp = config.unsupervised_train
        supervised_path = Path(str(config.output_path) + "GENERATED.pkl")
        config.unsupervised_train = tmp
        assert supervised_path.is_file()
        with open(str(supervised_path), "rb") as input_file:
            checkpoint = pickle.load(input_file)
            deep_nmf_original = checkpoint["deep_nmf"]

        optimizer = optim.Adam(deep_nmf_original.parameters(), lr=config.lr)
        unsupervised_path = Path(str(config.output_path) + "GENERATED-UNsup_new_loss.pkl")
        unsupervised_path_best = Path(str(config.output_path) + "GENERATED-UNsupB_new_loss.pkl")
        if unsupervised_path.is_file() and unsupervised_path_best.is_file():
            print(f"UnSupervised+generated file exsists {supervised_index}")
        else:
            print(f'start unsupervised {supervised_index}: : {datetime.now().strftime("%d-%m, %H:%M:%S")}')
            loss34 = train_with_generated_data_unsupervised(
                ref_mat, original_ref, mix_max, deep_nmf_original, optimizer, config, unsupervised_path_best, cells
            )
            checkpoint = {"deep_nmf": deep_nmf_original, "config": config, "loss34": loss34}
            if not unsupervised_path.is_file():
                with open(unsupervised_path, "wb") as f:
                    pickle.dump(checkpoint, f)

    learner = UnsupervisedLearner(config, deep_nmf_original, optimizer, mix_max, dist_mix_i, h_0_train, ref_mat)
    return learner


def handle_best(learner: UnsupervisedLearner) -> None:
    config = learner.config
    dist_mix_i = learner.dist_mix_i
    out, out34, deep_nmf, best_34_obj, best_i = _train_unsupervised(learner)
    _handle_final_matrix(config, out, out34, deep_nmf, dist_mix_i)

    dist_new = dist_mix_i.copy()
    dist_new[:] = best_34_obj.cpu().detach().numpy()
    i = config.unsupervised_train
    config.unsupervised_train = best_i
    dist_new.to_csv(config.output_path, sep="\t")
    config.unsupervised_train = i


def _find_loss(
    out: tensor,
    dist_mix_i_tensor: tensor,
) -> Tuple[tensor, float, tensor]:
    criterion = nn.MSELoss(reduction="mean")
    loss2 = torch.sqrt(criterion(out, dist_mix_i_tensor))

    out34 = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12))
    loss34 = torch.sqrt(criterion(out34, dist_mix_i_tensor))

    return loss2, loss34.item(), out34


def _train_unsupervised(
    learner: UnsupervisedLearner,
) -> Tuple[tensor, tensor, UnsuperNet, tensor, int, int, List[float], List[float]]:
    deep_nmf = learner.deep_nmf
    config = learner.config
    mix_max = _tensoring(learner.mix_max).to(config.device)
    optimizer = learner.optimizer
    dist_mix_i_tensor = _tensoring(learner.dist_mix_i.to_numpy()).to(config.device)
    h_0 = learner.h_0.to(config.device)

    inputs = (h_0, mix_max)
    torch.autograd.set_detect_anomaly(True)
    out = ndarray([])
    out34 = ndarray([])
    best_34 = 2
    best_34_obj = out
    best_i = 0
    loss = None
    nnls_result = [nnls(learner.ref, mix_max[kk])[0] for kk in range(len(mix_max))]
    nnls_result = _tensoring(np.asanyarray([d / sum(d) for d in nnls_result])).to(config.device)
    supervised_losses = []
    unsupervised_losses = []
    for i in range(config.unsupervised_train):
        out, loss = deep_nmf(*inputs)

        criterion = nn.MSELoss(reduction="mean")
        loss22 = criterion(out, nnls_result)

        loss = loss / mix_max.shape[1]
        features = mix_max.shape[1]
        w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        loss = cost_tns(mix_max, dnmf_w, out, config.l1_regularization, config.l2_regularization) / mix_max.shape[1]
        total_loss = loss
        unsupp_loss = 0.5 * torch.pow(mix_max - out.mm(dnmf_w), 2).sum() / (mix_max.shape[0] * mix_max.shape[1])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            loss2, loss34, out34 = _find_loss(out, dist_mix_i_tensor)
        if loss34 < best_34:
            best_34_obj = out
            best_34 = loss34
            best_i = i
        unsupervised_losses.append(unsupp_loss.detach().numpy())
        supervised_losses.append(loss34)
        if i % 500 == 0:
            print(
                f"i = {i}, loss =  {loss.item()} loss nnls {loss22} , total {total_loss} loss mix {loss * mix_max.shape[1]} loss criterion = {loss2} loss34 is {loss34} best {best_34}::{best_i}"
            )
            print(f'{datetime.now().strftime("%d-%m, %H:%M:%S")}')
    print(f"Finish: best {best_34}::{best_i}")
    return out, out34, deep_nmf, best_34_obj, best_i, loss.item(), unsupervised_losses, supervised_losses


def train_with_generated_data_unsupervised(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNet,
    optimizer: Optimizer,
    config: DnmfConfig,
    pickle_path: Path,
    cells: List[str],
) -> float:
    torch.autograd.set_detect_anomaly(True)
    rows, genes = mix_max.shape
    best_34 = 2
    best_i = 0
    for train_index in range(1, config.unsupervised_train + 1):
        generated_mix, generated_dist = generate_dists(original_ref_mat.T, train_index * 0.0001, rows, cells)
        mix_max = _quantile_normalize(generated_mix, original_ref_mat)
        _, mix_max = _normalize_zero_one(original_ref_mat, mix_max)

        generated_dist = _tensoring(generated_dist).to(config.device)
        mix_max = mix_max.T
        h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
        h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(config.device)

        mix_max = _tensoring(mix_max).to(config.device)
        out, loss = deep_nmf(h_0_train, mix_max)

        features = mix_max.shape[1]
        w_arrays = [nnls(out.data.numpy(), mix_max.T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        loss = cost_tns(mix_max, dnmf_w, out, config.l1_regularization, config.l2_regularization)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)

        with torch.no_grad():
            loss2, loss34, out34 = _find_loss(out, generated_dist)
        if train_index > 4000 and 0.125 < loss34 < 2:
            print(f"break: i = {train_index}")
            return loss34
        if loss34 < best_34:
            best_34 = loss34
            best_i = train_index
            checkpoint = {"deep_nmf": deep_nmf, "config": config, "loss34": loss34}
            with open(str(pickle_path), "wb") as f:
                pickle.dump(checkpoint, f)

        if train_index % 4000 == 0:
            print(
                f"i = {train_index}, loss =  {loss.item()} loss criterion = {loss2} loss34 is {loss34} best {best_34}::{best_i}"
            )
            print(f'{datetime.now().strftime("%d-%m, %H:%M:%S")}')
    return loss34


def train_with_generated_data(
    ref_mat: ndarray,
    original_ref_mat: ndarray,
    mix_max: ndarray,
    deep_nmf: UnsuperNet,
    optimizer: Optimizer,
    config: DnmfConfig,
    cells: List[str],
):
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
        if train_index % 8000 == 0:
            print(
                f"{datetime.now().strftime('%d-%m, %H:%M:%S')}: Train: Start train index: {train_index} with loss: {loss_values}"
            )
        if (train_index % SUPERVISED_SPLIT == 0 and train_index != 0) or (train_index == supervised_train):
            config.supervised_train = train_index
            checkpoint = {"deep_nmf": deep_nmf, "config": config}
            with open(str(config.output_path) + "GENERATED.pkl", "wb") as f:
                pickle.dump(checkpoint, f)
            config.supervised_train = supervised_train + 1


def train_supervised_one_sample(
    ref_mat: ndarray, mix_max: ndarray, dist_mat: tensor, deep_nmf: UnsuperNet, optimizer: Optimizer, device
):
    h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
    # h_0_train = [nnls(ref_mat, mix_max[kk])[0] for kk in range(len(mix_max))]
    h_0_train = _tensoring(np.asanyarray([d / sum(d) for d in h_0_train])).to(device)

    criterion = nn.MSELoss(reduction="mean")
    # Train the Network
    mix_max = _tensoring(mix_max).to(device)
    out, _ = deep_nmf(h_0_train, mix_max)
    loss = torch.sqrt(criterion(out, dist_mat))  # loss between predicted and truth
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for w in deep_nmf.parameters():
        w.data = w.clamp(min=0, max=math.inf)
    return loss.item()

from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import entr

from utils.data_frame_utils import get_shared_indexes

ZERO_INCOMPLETE_SIZE = 0.5


def run_gedit_preprocessing(
    mix_max: DataFrame, ref_mat: DataFrame, total_sigs: int, use_all_genes: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ref_mat = ref_mat.loc[(ref_mat != 0).any(axis=1)]
    mix_max[:] = _quantile_normalize(mix_max.to_numpy(), ref_mat.to_numpy())
    share_mix, share_ref = get_shared_indexes(mix_max, ref_mat)
    if not use_all_genes:
        sig_ref = _select_sub_genes_entropy(share_ref, total_sigs)
        share_mix, share_ref = get_shared_indexes(mix_max, sig_ref)
    ref, mix = _normalize_zero_one(share_ref.to_numpy(), share_mix.to_numpy())
    return ref, mix, share_ref.index.to_numpy()


def _quantile_normalize(mix_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
    flatten_ref = ref_data.flatten()
    flatten_ref.sort()

    sorted_indexes_mix = mix_data.T.argsort().argsort()
    sorted_indexes_mix = ((sorted_indexes_mix / ((mix_data.shape[0]) - 1)) * (len(flatten_ref) - 1)).astype(int)
    return flatten_ref[sorted_indexes_mix].T


def _select_sub_genes_entropy(ref_mat: DataFrame, total_sigs: int) -> DataFrame:
    ref_mat = ref_mat.loc[(ref_mat == 0).mean(axis=1) <= ZERO_INCOMPLETE_SIZE]
    all_sigs = total_sigs * ref_mat.shape[1]
    min_value = ref_mat[ref_mat != 0].min().min()
    ref_mat = ref_mat.replace(to_replace=0, value=min_value)
    normalize_ref = (ref_mat.T / ref_mat.sum(axis=1)).T
    entropy = -1 * entr(normalize_ref).sum(axis=1)
    max_index = ref_mat.idxmax(axis=1)
    entropy_max_index = pd.concat([entropy, max_index], axis=1)
    entropy_max_index.columns = ["entropy", "cell"]
    genes = (
        entropy_max_index.groupby(["cell"])
        .apply(lambda x: x.nlargest(total_sigs, columns="entropy"))["entropy"]
        .reset_index()
        .iloc[:, 1]
    )
    left_gens = all_sigs - len(genes)
    if left_gens > 0 and entropy_max_index.shape[0] > len(genes):
        other_genes = entropy_max_index.drop(genes).nlargest(left_gens, columns="entropy").reset_index().iloc[:, 0]
        genes = pd.concat([genes, other_genes])
    return ref_mat.loc[genes]


def _normalize_zero_one(ref_mat: np.ndarray, mix_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    merged_mat = np.concatenate((ref_mat, mix_mat), axis=1)
    min_merged = merged_mat.min(axis=1)
    max_merged = merged_mat.max(axis=1)
    ref_mat = ((ref_mat.T - min_merged) / (max_merged - min_merged)).T
    mix_mat = ((mix_mat.T - min_merged) / (max_merged - min_merged)).T
    return ref_mat, mix_mat

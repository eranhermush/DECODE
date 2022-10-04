from typing import Tuple, Optional

from pandas import DataFrame, Index

from DECODE.utils.data_frame_utils import get_shared_indexes

OTHER_NAME = "Other"


def format_dataframe(
    ref_mat: DataFrame, mix_dist: DataFrame, mix_mat: Optional[DataFrame] = None
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Assumes that the dimensions are:
    ref_mat: genes * cells
    mix_dist: samples * cells
    mix_mat: genes * samples
    """
    ref_mat = ref_mat.rename(_convert_name, axis="columns")
    mix_dist = mix_dist.rename(_convert_name, axis="columns")
    ref_mat = ref_mat.groupby(level=0, axis=1).sum()
    mix_dist = mix_dist.groupby(level=0, axis=1).sum()
    _, ref_mat = get_shared_indexes(mix_dist.T, ref_mat.T)
    if OTHER_NAME in ref_mat.index:
        ref_mat = ref_mat.drop(OTHER_NAME)
    ref_mat = ref_mat.T
    difference_cells = mix_dist.columns.difference(ref_mat.columns)
    if OTHER_NAME in mix_dist.T.index and OTHER_NAME not in difference_cells:
        difference_cells.append(Index([OTHER_NAME]))
    mix_dist = mix_dist.drop(difference_cells, axis=1)
    assert mix_dist.columns.tolist() == ref_mat.columns.tolist()
    return ref_mat, mix_dist, mix_mat


def _convert_name(cell_name: str) -> str:
    cell_name = cell_name.lower()
    if "endothelial" in cell_name or "endo" in cell_name:
        return "Endothelial"
    if "neutrophil" in cell_name:
        return "Neutrophils"
    if "macrophage" in cell_name:
        return "Macrophages"
    if "fibroblast" in cell_name or "cafs" in cell_name:
        return "Fibroblasts"
    if "cd4" in cell_name:
        return "CD4 T Cells"
    if "cd8" in cell_name:
        return "CD8 T Cells"
    if "mono" in cell_name or "cd14" in cell_name:
        return "Monocytes"
    if "nk" in cell_name or "natural" in cell_name:
        return "NK Cells"
    if "mast" in cell_name:
        return "Mast Cell"
    if (
        cell_name == "b"
        or "cd19b" in cell_name
        or "b cell" in cell_name
        or "b_cell" in cell_name
        or "b.cell" in cell_name
        or "bcell" in cell_name
        or "b lineage" in cell_name
        or "b-cell" in cell_name
        and "pro" not in cell_name
    ):
        return "B Cells"
    if cell_name in ["p-value", "correlation", "rmse", "absolute score (sig.score)"]:
        return "trash"
    return OTHER_NAME

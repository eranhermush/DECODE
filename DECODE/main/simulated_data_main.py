import os
from pathlib import Path

import torch

from preprocessing.decode_config import DecodeConfig
from train.train import train_manager


def run_main_simulated(ref_folder, output_folder, mix_folder, dist_folder, index):
    """
    Run training on simulated data - supervised and then unsupervised trainings
    :param output_folder: In this folder we save pickle files of our model (to future use, it can be a tmp folder)
    :param ref_folder: path to the folder of the signature matrix
    :param mix_folder: path to the folder of the datasets
    :param dist_folder: path to the folder of the matrices that define the final fractions
        (for each dataset 'x.tsv' we have 'TruePropsx.tsv' file with one row - the cells to predict)
    :param index: index of our dataset
    """
    torch.autograd.set_detect_anomaly(True)

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))

    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))
    mix = mixes[index]
    for ref_name in refs:
        mix_p = Path(mix)
        dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"
        config = DecodeConfig(
            use_gedit=True,
            use_w0=True,
            output_folder=Path(output_folder),
            unsup_output_folder=Path(output_folder),
            ref_path=Path(ref_name),
            mix_path=mix_p,
            dist_path=dist_path,
            num_layers=4,
            supervised_train=60000,
            rewrite_exists_output=True,
            l1_regularization=0,
            l2_regularization=0,
            total_sigs=50,
        )
        train_manager(config)

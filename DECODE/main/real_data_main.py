import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from preprocessing.decode_config import DecodeConfig

from train.train import train_manager, train_unsupervised_real_data


def run_main_real_data(ref_folder, output_folder, mix_folder, dist_folder, index, output_folder_for_results=None):
    """
    Run the final step of DECODE - unsupervised training on real data
    :param ref_folder: path to the folder of the signature matrix
    :param output_folder: In this folder we use the pickle files of the training on simulated data
    :param mix_folder: path to the folder of the datasets
    :param dist_folder: path to the folder of the matrices that define the final fractions
        (for each dataset 'x.tsv' we have 'TruePropsx.tsv' file with one row - the cells to predict)
    :param index: index of our dataset
    :param output_folder_for_results: Folder of the output result - final prediction
    :return:
    """
    if output_folder_for_results is None:
        output_folder_for_results = output_folder
    torch.autograd.set_detect_anomaly(True)

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))

    mix = mixes[index]
    best_loss_ref = 200
    best_nmf_ref = None

    for ref_name in refs:
        best_loss = 200
        learner_best_supervised = None

        mix_p = Path(mix)
        dist_path = Path(dist_folder) / f"TrueProps{mix_p.name}"

        config = DecodeConfig(
            use_gedit=True,
            use_w0=True,
            output_folder=Path(output_folder),
            unsup_output_folder=Path(output_folder_for_results),
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
        print(config.full_str())
        unsupervised_path = Path(str(config.output_path) + "_GENERATED-UNsup_new_loss_algo.pkl")
        if unsupervised_path.is_file():
            with open(str(unsupervised_path), "rb") as input_file:
                checkpoint = pickle.load(input_file)
            loss = checkpoint["normalize_loss"]
            if loss < best_loss:
                best_loss = loss
                best_alg = checkpoint["deep_nmf"]
                learner_best_supervised = train_manager(config, False)
                learner_best_supervised.deep_nmf = best_alg
                learner_best_supervised.optimizer = optim.Adam(best_alg.parameters(), lr=config.lr)
        else:
            print(f"doesnt have {unsupervised_path} file")
        if learner_best_supervised is not None:
            for iteration_size in [101]:
                learner_best_supervised.config.supervised_train = iteration_size
                print(f"best is {learner_best_supervised.config.full_str()}")
                _, normalize_matrix_out, _, _, _, loss = train_unsupervised_real_data(learner_best_supervised)
                if loss < best_loss_ref:
                    best_loss_ref = loss
                    best_nmf_ref = learner_best_supervised, normalize_matrix_out

    dist_new = best_nmf_ref[0].dist_mix_i.copy()
    dist_new = pd.DataFrame(best_nmf_ref[1].cpu().detach().numpy(), columns=dist_new.columns)
    dist_new.to_csv(best_nmf_ref[0].config.unsup_output_path, sep="\t")
    return

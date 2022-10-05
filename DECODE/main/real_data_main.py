import os
import pickle
from pathlib import Path

import torch
from torch import optim

from DECODE.preprocessing.dnmf_config import DnmfConfig
from DECODE.train.train import train_manager, _train_unsupervised


def run_main_real_data(ref_folder, output_folder, mix_folder, dist_folder, index, output_folder_for_results=None):
    if output_folder_for_results is None:
        output_folder_for_results = output_folder
    torch.autograd.set_detect_anomaly(True)
    num_layers_options = [4]
    supervised_trains = [20000, 40000, 60000]

    refs = []
    for filename in os.listdir(ref_folder):
        refs.append(os.path.join(ref_folder, filename))
    mixes = []
    for filename in os.listdir(mix_folder):
        mixes.append(os.path.join(mix_folder, filename))

    mix = mixes[index]
    best_loss_ref = 200
    best_nmf_ref = None

    best_of_best_loss_ref = 200
    best_of_best_nmf_ref = None

    for ref_name in refs:
        best_of_best_loss = 200
        best_loss = 200
        learner_best1 = None
        learner_best = None

        mix_p = Path(mix)
        dist_path = Path(dist_folder) / f"{mix_p.name}"

        for num_layers_option in num_layers_options:
            for supervised_train in supervised_trains:
                config = DnmfConfig(
                    use_gedit=True,
                    use_w0=True,
                    w1_option="algo",
                    output_folder=Path(output_folder),
                    unsup_output_folder=Path(output_folder),
                    ref_path=Path(ref_name),
                    mix_path=mix_p,
                    dist_path=dist_path,
                    num_layers=num_layers_option,
                    supervised_train=supervised_train,
                    unsupervised_train=20000,
                    l1_regularization=0,
                    l2_regularization=0,
                    rewrite_exists_output=True,
                    total_sigs=50,
                    lr=0.002,
                )
                print(config.full_str())
                unsupervised_path = Path(str(config.output_path) + "GENERATED-UNsup_new_loss.pkl")
                unsupervised_path_best = Path(str(config.output_path) + "GENERATED-UNsupB_new_loss.pkl")
                if unsupervised_path.is_file():
                    with open(str(unsupervised_path), "rb") as input_file:
                        checkpoint = pickle.load(input_file)
                    loss = checkpoint["loss34"]
                    if loss < best_loss:
                        best_loss = loss
                        best_alg = checkpoint["deep_nmf"]
                        learner_best1 = train_manager(config, False)
                        learner_best1.deep_nmf = best_alg
                        learner_best1.optimizer = optim.Adam(best_alg.parameters(), lr=config.lr)
                else:
                    print(f"doesnt have {unsupervised_path} file")
                if unsupervised_path_best.is_file():
                    with open(str(unsupervised_path_best), "rb") as input_file:
                        checkpoint = pickle.load(input_file)
                    loss = checkpoint["loss34"]
                    if loss < best_of_best_loss:
                        best_of_best_loss = loss
                        best_of_best_alg = checkpoint["deep_nmf"]
                        learner_best = train_manager(config, False)
                        learner_best.deep_nmf = best_of_best_alg
                        learner_best.optimizer = optim.Adam(best_of_best_alg.parameters(), lr=config.lr)
                else:
                    print(f"doesnt have {unsupervised_path_best} file")
            if learner_best1 is not None:
                for unsupervised_train_size in [101]:
                    learner_best1.config.unsupervised_train = unsupervised_train_size
                    learner_best.config.unsupervised_train = unsupervised_train_size
                    print(f"best is {learner_best1.config.full_str()}")
                    print(f"best is {learner_best.config.full_str()}")
                    _, out34, _, _, _, loss, _, _ = _train_unsupervised(learner_best1)
                    if loss < best_loss_ref:
                        best_loss_ref = loss
                        best_nmf_ref = learner_best1, out34

                    _, out34, _, _, _, loss, _, _ = _train_unsupervised(learner_best)
                    if loss < best_of_best_loss_ref:
                        best_of_best_loss_ref = loss
                        best_of_best_nmf_ref = learner_best, out34

    if best_of_best_loss_ref < best_loss_ref:
        dist_new = best_of_best_nmf_ref[0].dist_mix_i.copy()
        dist_new[:] = best_of_best_nmf_ref[1].cpu().detach().numpy()
        dist_new.to_csv(best_of_best_nmf_ref[0].config.unsup_output_path, sep="\t")
        return

    dist_new = best_nmf_ref[0].dist_mix_i.copy()
    dist_new[:] = best_nmf_ref[1].cpu().detach().numpy()
    dist_new.to_csv(best_nmf_ref[0].config.unsup_output_path, sep="\t")

    return

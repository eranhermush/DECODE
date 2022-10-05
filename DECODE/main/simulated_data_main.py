import os
from pathlib import Path

import torch

from DECODE.preprocessing.dnmf_config import DnmfConfig
from DECODE.train.train import SUPERVISED_SPLIT, train_manager


def run_main_simulated(ref_folder, output_folder, mix_folder, dist_folder, index):
    torch.autograd.set_detect_anomaly(True)
    num_layers_options = [4]
    supervised_trains = [3 * SUPERVISED_SPLIT]
    total_sigs = [50]

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
        for num_layers_option in num_layers_options:
            for supervised_train in supervised_trains:
                for total_sig in total_sigs:
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
                        rewrite_exists_output=True,
                        l1_regularization=1,
                        total_sigs=total_sig,
                        lr=0.002,
                    )
                    print(f"Main, {config.full_str()}")
                    train_manager(config)

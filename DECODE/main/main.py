from DECODE.main.real_data_main import run_main_real_data
from DECODE.main.simulated_data_main import run_main_simulated


def main(
    output_folder: str,
    ref_folder: str,
    mixes_folder: str,
    true_prop_folder: str,
    output_folder_final: str,
    index_dataset: int,
):
    run_main_simulated(ref_folder, output_folder, mixes_folder, true_prop_folder, index_dataset)
    run_main_real_data(ref_folder, output_folder, mixes_folder, true_prop_folder, index_dataset, output_folder_final)

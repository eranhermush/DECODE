from DECODE.main.real_data_main import run_main_real_data
from DECODE.main.simulated_data_main import run_main_simulated


def main(
    output_folder: str,
    ref_folder: str,
    mixes_folder: str,
    true_prop_folder: str,
    output_folder_final: str,
    index_dataset: int = 0,
):
    run_main_simulated(ref_folder, output_folder, mixes_folder, true_prop_folder, index_dataset)
    run_main_real_data(ref_folder, output_folder, mixes_folder, true_prop_folder, index_dataset, output_folder_final)


if __name__ == "__main__":
    output_folder = "<in this folder we will save the models after training with simulated data>"
    output_folder_final = "<in this folder we will save the final models>"
    ref_folder = "<path to signature matrix folder>"
    dist_path = (
        "<path to folder with files for each dataset, each file needs to contain one column with the desired cells>"
    )
    mix_folder = "<path to folder with datasets>"
    main(output_folder, ref_folder, mix_folder, dist_path, output_folder_final)

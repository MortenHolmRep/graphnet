"""Running inference on GraphSAGE-cleaned pulses in IceCube-Upgrade."""

import logging
from os import makedirs

import torch

from graphnet.utilities.logging import get_logger

logger = get_logger(logging.DEBUG)

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Main function definition
def main(
    input_path: str,
    output_path: str,
    model_path: str,
):

    # Make sure output directory exists
    makedirs(dirname(output_path), exist_ok=True)

    device = torch.device("cpu")
        
    # Load model
    model = torch.load(model_path)
    model.to(device)
    model.eval()


# Main function call
if __name__ == "__main__":

    input_db = "/groups/icecube/peter/storage/Sebastian_MoonDataL4.db"
    output_folder = "/groups/icecube/qgf305/storage/test/Saskia_datapipeline/L2_2018_1/"
    model_path = "/groups/icecube/peter/storage/test/dev_lvl7_robustness_muon_neutrino_0000/dynedge_zenith_example/dynedge_zenith_example_model.pth"

    main(input_db, output_folder, model_path)

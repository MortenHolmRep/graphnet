"""Minimum working example (MWE) to use SQLiteDataConverter."""

import logging
import os

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
    I3GCDExtractor,
)
from graphnet.data.sqlite.sqlite_dataconverter import SQLiteDataConverter

logger = get_logger(level=logging.DEBUG)


def main_icecube86():
    """Main script function."""
    paths = [
        "/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/moonL4_segspline_exp13_01_redo_00001.i3.bz2"
    ]
    gcd_extractor = I3GCDExtractor("00001")
    gcd_extractor.set_files(paths)
    gcd_rescue = gcd_extractor(paths)
    pulsemap = "SRTInIcePulses"
    outdir = "/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/data_out"
    

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(pulsemap),
        ],
        outdir,
        gcd_rescue,
    )
    converter(paths)


def main_icecube_upgrade():
    """Main script function."""
    basedir = "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/"
    paths = [os.path.join(basedir, "step4")]
    outdir = "/groups/icecube/asogaard/temp/sqlite_test_upgrade"
    workers = 10

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_mDOM"
            ),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_DEgg"
            ),
        ],
        outdir,
        workers=workers,
        nb_files_to_batch=1000,
        # sequential_batch_pattern="temp_{:03d}",
        input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        verbose=1,
    )
    converter(paths)


if __name__ == "__main__":
    main_icecube86()
    # main_icecube_upgrade()

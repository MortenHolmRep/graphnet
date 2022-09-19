"""Minimum working example (MWE) to use SQLiteDataConverter."""

import logging
import os

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.sqlite.sqlite_dataconverter import SQLiteDataConverter

logger = get_logger(level=logging.DEBUG)


def main_icecube86():
    """Main script function."""
    paths = [
        "/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/moonL4_segspline_exp13_01_redo_00001.i3.bz2"
    ]
    pulsemap = "SRTInIcePulses"
    gcd_rescue = "/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/Level2_IC86.2012_data_Run00121480_0101_GCD.i3.gz"
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
    gcd_rescue = os.path.join(
        basedir, "gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
    )
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
        gcd_rescue,
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

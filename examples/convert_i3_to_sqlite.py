"""Minimum working example (MWE) to use SQLiteDataConverter
"""

from graphnet.data.i3extractor import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.sqlite_dataconverter import SQLiteDataConverter


def main_icecube86():
    """Main script function."""
    paths = [
         "/data/user/mholm/data/I3_files/"
        #"/data/ana/LE/oscNext/pass2/genie/level7_v03.01/140000"
    ]
    pulsemap = "SRTInIcePulses"
    gcd_rescue = "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
    outdir = "/data/user/mholm/data/sqlite_test_ic86"
    db_name = "data_test"
    workers = 1

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(pulsemap),
        ],
        outdir,
        gcd_rescue,
        db_name=db_name,
        workers=workers,
    )
    converter(paths)


def main_icecube_upgrade():
    """Main script function."""
    paths = [
        "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4"
    ]
    gcd_rescue = (
        "resources/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
    )
    outdir = "/groups/icecube/asogaard/temp/sqlite_test_upgrade"
    db_name = "data_test"
    workers = 5

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
        db_name=db_name,
        workers=workers,
        verbose=1,
    )
    converter(paths)


if __name__ == "__main__":
    main_icecube86()
    #main_icecube_upgrade()

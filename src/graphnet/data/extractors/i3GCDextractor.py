from abc import ABC, abstractmethod
from typing import List

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import LoggerMixin

if has_icecube_package():
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class I3GCDExtractor(ABC, LoggerMixin):
    """Extracts relevant information from geometry frames as the gcd reference, from an I3 file."""

    def __init__(self, name):

        # Member variables
        self._i3_file = None
        self._gcd_dict = None
        self._name = name

    def set_files(self, i3_file):
        self._i3_file = i3_file
        self._load_gcd_data()

    def _load_gcd_data(self):
        """Loads the geospatial information contained in the geometry frame of the i3 file."""
        i3_file = dataio.I3File(self._i3_file)
        g_frame = i3_file.pop_frame(icetray.I3Frame.Geometry)
        self._gcd_dict = g_frame["I3Geometry"].omgeo

# desi_utils.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Set, List

import os
import shutil
import numpy as np
import astropy.units as u
from astropy.table import QTable, Table
import agama

# Your existing utils
from utils import mock_stream_utils, coordinate_utils, selection_utils, plot_utils

# ---------------------------
# GC parameters & potentials
# ---------------------------

@dataclass
class GCParams:
    """Helper to load the GC parameter table and select a cluster row."""
    table_path: str = "data/mw_gc_parameters_orbital_structural_time.ecsv"

    def get_row(self, cluster_name: str) -> QTable:
        tab = QTable.read(self.table_path)
        sel = tab[tab["Cluster"] == cluster_name]
        if len(sel) == 0:
            raise ValueError(f"Cluster '{cluster_name}' not found in: {self.table_path}")
        return sel[0:1]  # keep as 1-row table for convenience


class PotentialFactory:
    """Create host & satellite potentials, and ensure Agama unit setup."""
    def __init__(self, potentials_dir: str = "potentials"):
        self.potentials_dir = potentials_dir
        self._ensure_units()

    @staticmethod
    def _ensure_units():
        # Matches your original: code assumes ini files are already in these units
        agama.setUnits(length=1, velocity=1, mass=1)

    def host(self, potential_name: str) -> agama.GalaPotential:
        ini = Path(self.potentials_dir) / f"{potential_name}.ini"
        if not ini.is_file():
            raise FileNotFoundError(f"Host potential ini not found: {ini}")
        return agama.GalaPotential(str(ini))

    @staticmethod
    def satellite_plummer(mass: float, rhm: float) -> agama.GalaPotential:
        # scaleRadius expects same length units as Agama config (here unitless, consistent with setUnits above)
        return agama.GalaPotential(type="Plummer", mass=mass, scaleRadius=rhm)

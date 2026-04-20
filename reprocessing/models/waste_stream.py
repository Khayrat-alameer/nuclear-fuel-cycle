"""
Waste Stream Models
=================
Secondary waste generation and characterization in reprocessing.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class WasteParameters:
    """Parameters for waste stream analysis."""

    feed_assay_u235: float = 0.03  # 3% enriched
    burnup: float = 50.0  # GWd/tU
    cooling_time: float = 5.0  # years
    recovery_efficiency: float = 0.99  # 99% recovery


class WasteStreamModel:
    """
    Model waste streams from reprocessing operations.
    """

    def __init__(self, params: WasteParameters):
        self.params = params

    def calculate_fission_products(self) -> Dict[str, float]:
        """
        Calculate fission product inventory based on burnup.
        Based on ORIGEN calculations.
        """
        # Simplified fission product yields
        burnup_factor = self.params.burnup / 50.0  # Normalize to 50 GWd/tU

        return {
            "cs137": 33.0 * burnup_factor,  # PBq/tHM
            "sr90": 15.0 * burnup_factor,
            "ru106": 5.0 * burnup_factor,
            "zr95": 8.0 * burnup_factor,
            "nb95": 8.0 * burnup_factor,
            "ce144": 12.0 * burnup_factor,
            "total_alpha": 0.5 * burnup_factor,  # TBq/tHM
        }

    def calculate_actinides(self) -> Dict[str, float]:
        """Calculate remaining actinides."""
        p = self.params
        rec = p.recovery_efficiency

        return {
            "u238": 1000 * rec,  # kg/tHM (mostly recovered)
            "u235": 30 * (1 - rec),  # Small loss
            "pu239": 10 * (1 - rec),  # Partial recovery
            "pu240": 4 * (1 - rec),
            "am241": 0.5,
            "cm244": 0.1,
        }

    def get_waste_volume(self) -> Dict[str, float]:
        """Estimate waste volumes from reprocessing."""
        # HLW: 200-400 L/tHM
        # ILW: 50-100 L/tHM
        # LLW: 10-20 m3/tHM

        return {"hlw_liters": 300.0, "ilw_liters": 75.0, "llw_m3": 15.0}

    def get_waste_classification(self) -> str:
        """Determine primary waste classification."""
        fp = self.calculate_fission_products()

        if fp["total_alpha"] > 0.1:
            return "HLW (High Level Waste)"
        elif fp["sr90"] + fp["cs137"] > 10:
            return "ILW (Intermediate Level Waste)"
        else:
            return "LLW (Low Level Waste)"

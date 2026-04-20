"""
Separation Chemistry Models for Reprocessing
==========================================
Solvent extraction chemistry and mass balance calculations.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class SolventExtractionParams:
    """Parameters for solvent extraction."""

    tbp_concentration: float = 0.30  # 30% TBP in dodecane
    aqueous_flow: float = 100.0  # L/h
    organic_flow: float = 100.0  # L/h
    nitric_acid: float = 2.0  # M
    temperature: float = 25.0  # Celsius


class SeparationChemistryModel:
    """
    Solvent extraction chemistry for uranium/plutonium separation.
    """

    def __init__(self, params: SolventExtractionParams):
        self.params = params

    def calculate_distribution_ratio(self, metal: str, concentration: float) -> float:
        """
        Calculate distribution ratio D for given metal.

        Based on PUREX chemistry from CEA data.
        """
        p = self.params

        # Uranium extraction
        if metal.lower() == "uranium":
            # D_U = f([HNO3], [TBP], [U])
            D = 0.4 * (p.tbp_concentration**1.5) * (p.nitric_acid**-0.5)
            D *= 1 + 0.02 * concentration
            return D

        # Plutonium extraction
        elif metal.lower() == "plutonium":
            D = 0.8 * (p.tbp_concentration**2.0) * (p.nitric_acid**-0.3)
            return D

        # Fission products (lower extraction)
        else:
            return 0.01 * (p.tbp_concentration**0.5)

    def simulate_stage(
        self, aqueous_in: float, organic_in: float, metal: str = "uranium"
    ) -> Tuple[float, float]:
        """
        Simulate single extraction stage equilibrium.

        Returns: (aqueous_out, organic_out)
        """
        D = self.calculate_distribution_ratio(metal, aqueous_in)

        # Equilibrium calculation
        total = aqueous_in + organic_in
        aqueous_out = total / (1 + D)
        organic_out = total * D / (1 + D)

        return aqueous_out, organic_out

    def calculate_separation_factor(self, metal1: str, metal2: str) -> float:
        """Calculate separation factor between two metals."""
        D1 = self.calculate_distribution_ratio(metal1, 1.0)
        D2 = self.calculate_distribution_ratio(metal2, 1.0)

        if D2 <= 0:
            return float("inf")
        return D1 / D2

    def get_mass_balance(self, feed_u: float, feed_pu: float) -> Dict[str, float]:
        """Calculate mass balance for U/Pu separation."""
        # Extract stages
        aq_u, org_u = self.simulate_stage(feed_u, 0, "uranium")
        aq_pu, org_pu = self.simulate_stage(feed_pu, 0, "plutonium")

        # Scrub stages (remove entrained impurities)
        # Strip stages (recover products)

        return {
            "u_organic": org_u,
            "u_aqueous": aq_u,
            "pu_organic": org_pu,
            "pu_aqueous": aq_pu,
            "u_pu_separation": self.calculate_separation_factor("uranium", "plutonium"),
        }

"""
PUREX Process Models
==================
Spent fuel reprocessing using Plutonium Uranium Reduction Extraction.
Based on research from CEA PAREX code and recent papers (2011-2024).
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PUREXParameters:
    """Parameters for PUREX process simulation."""

    feed_flow_rate: float = 100.0  # L/h
    aqueous_acid_conc: float = 2.0  # M HNO3
    organic_acid_conc: float = 0.0  # M (TBP in organic)
    scrub_acid_conc: float = 0.5  # M HNO3
    strip_acid_conc: float = 0.01  # M HNO3
    extraction_stages: int = 12
    scrub_stages: int = 4
    strip_stages: int = 6


class PUREXModel:
    """
    PUREX solvent extraction model.

    Simulates uranium/plutonium recovery from spent fuel using
    TBP-based liquid-liquid extraction.
    """

    def __init__(self, params: PUREXParameters):
        self.params = params
        self.results = {}

    def _distribution_coefficient(self, acid_conc: float, metal_conc: float) -> float:
        """Calculate distribution coefficient D for uranium."""
        # Based on PUREX technical manual
        # D = A * [H+]^(-B) relation
        A = 0.5
        B = 0.7
        if acid_conc <= 0:
            return 0.0
        return A * (acid_conc ** (-B)) * (1 + 0.1 * metal_conc)

    def simulate_extraction(self) -> Dict[str, np.ndarray]:
        """Simulate extraction, scrubbing, stripping cascade."""
        p = self.params
        n_stages = p.extraction_stages + p.scrub_stages + p.strip_stages

        # Stage-wise concentrations
        aqueous_u = np.zeros(n_stages)  # Uranium in aqueous
        organic_u = np.zeros(n_stages)  # Uranium in organic

        # Initialize feed concentration
        feed_u = 50.0  # g/L uranium in feed
        aqueous_u[-1] = feed_u

        # McCabe-Thiele stage-by-stage calculation
        # (simplified steady-state model)
        for stage in range(n_stages):
            D = self._distribution_coefficient(p.aqueous_acid_conc, aqueous_u[stage])

            # Extraction equilibrium
            if stage < p.extraction_stages:
                organic_u[stage] = D * aqueous_u[stage]
                aqueous_u[max(0, stage - 1)] = aqueous_u[stage]
            elif stage < p.extraction_stages + p.scrub_stages:
                # Scrubbing - remove impurities
                organic_u[stage] = 0.1 * D * aqueous_u[stage]
            else:
                # Stripping - recover uranium
                organic_u[stage] = 0.01 * aqueous_u[stage]
                aqueous_u[stage - 1] = feed_u * 0.5 ** (stage - p.extraction_stages)

        # Recovery calculations
        u_recovery = (1 - aqueous_u[0] / feed_u) if feed_u > 0 else 0

        return {
            "aqueous_uranium": aqueous_u,
            "organic_uranium": organic_u,
            "stages": n_stages,
            "u_recovery": u_recovery,
            "u_in_product": aqueous_u[0],
            "u_in_tailings": aqueous_u[-1],
        }

    def get_recovery_efficiency(self) -> Dict[str, float]:
        """Calculate recovery efficiencies."""
        sim = self.simulate_extraction()
        return {
            "uranium_recovery": sim["u_recovery"],
            "product_purity": sim["u_in_product"] / 50.0,
            "tailings_loss": sim["u_in_tailings"] / 50.0,
        }


class RecoveryEfficiencyModel:
    """Calculate overall recovery efficiency for reprocessing."""

    def __init__(self, params: PUREXParameters):
        self.params = params
        self.purex = PUREXModel(params)

    def calculate_recovery(self) -> Dict[str, float]:
        """Calculate metal recovery."""
        eff = self.purex.get_recovery_efficiency()
        return {
            "u_recovery": eff["uranium_recovery"],
            "product_assay": eff["product_purity"],
            "waste_generated": 1 - eff["uranium_recovery"],
        }

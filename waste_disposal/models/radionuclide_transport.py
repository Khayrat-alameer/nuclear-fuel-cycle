"""
Radionuclide Transport Models
========================
Migration through geological barriers.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class TransportParameters:
    """Radionuclide transport parameters."""

    solubility: float = 1e-6  # mol/L
    sorption_kd: float = 10.0  # m³/kg (distribution coefficient)
    diffusion_coeff: float = 1e-10  # m²/s
    porosity: float = 0.3


class RadionuclideTransportModel:
    """
    Radionuclide migration through geological barriers.
    """

    def __init__(self, params: TransportParameters):
        self.params = params

    def calculate_retardation_factor(self, nuclide: str) -> float:
        """
        Calculate retardation factor R = 1 + (rho_b * Kd) / theta

        Higher = slower migration
        """
        p = self.params
        rho = 2500  # kg/m3 (rock density)

        # Typical Kd values
        kd_values = {
            "cs137": 1.0,
            "sr90": 0.5,
            "u238": 50.0,
            "pu239": 100.0,
            "am241": 50.0,
        }

        Kd = kd_values.get(nuclide, 10.0)
        R = 1 + (rho * Kd) / p.porosity

        return R

    def calculate_migration_time(self, nuclide: str, distance: float) -> float:
        """Calculate time to migrate given distance."""
        p = self.params

        R = self.calculate_retardation_factor(nuclide)
        D = p.diffusion_coeff

        # t = (x² * R) / (2 * D)
        time_seconds = (distance**2 * R) / (2 * D)
        time_years = time_seconds / (365.25 * 86400)

        return time_years

    def calculate_concentration_profile(
        self, nuclide: str, distance: float
    ) -> np.ndarray:
        """Calculate concentration vs distance."""
        R = self.calculate_retardation_factor(nuclide)

        # Simplified exponential decay
        x = np.linspace(0, distance, 50)
        C = np.exp(-x / (100 * R))

        return C

    def calculate_annual_dose(self, nuclide: str) -> float:
        """Calculate annual dose from nuclide"""
        times = [100, 1000, 10000, 100000]
        doses = []

        for t in times:
            conc = 1.0 / (1 + t / 10000)  # Simplified
            # Dose conversion factor (simplified)
            dose_factor = 1e-11  # Sv/Bq
            dose = conc * 1e12 * dose_factor  # Approximate
            doses.append(dose)

        return np.max(doses)


class EngineeredBarrierModel:
    """Canister and buffer performance."""

    def calculate_corrosion_rate(self, material: str) -> float:
        """Calculate corrosion rate."""
        rates = {
            "carbon_steel": 10e-6,  # m/year
            "copper": 0.1e-6,
            "titanium": 0.5e-6,
        }
        return rates.get(material, 1e-6)

    def calculate_buffer_swelling(self, time: float) -> float:
        """Calculate buffer swelling over time."""
        # Bentonite swelling
        swelling = min(0.2, 0.01 * np.sqrt(time))
        return swelling

    def get_barrier_performance(self) -> Dict[str, float]:
        """Get barrier performance metrics."""
        return {
            "canister_integrity": 0.95,
            "buffer_porosity_change": 0.1,
            "radionuclide_sorption": 0.8,
        }

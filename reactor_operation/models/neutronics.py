"""
Neutronics Models
==============
Reactor physics and neutron transport.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class ReactorParameters:
    """Basic reactor parameters."""

    power: float = 3000.0  # MWthermal
    fuel_type: str = "UO2"
    enrichment: float = 0.035  # 3.5% U-235
    active_height: float = 4.0  # m
    active_radius: float = 1.8  # m


@dataclass
class NeutronicsParameters:
    """Neutronics calculation parameters."""

    core_power_density: float = 100.0  # kW/L
    k_effective: float = 1.0
    neutron_flux: float = 1e14  # n/cm2/s
    burnup: float = 0.0  # MWd/kgU


class NeutronicsModel:
    """
    Reactor neutronics calculations.
    """

    def __init__(self, params: ReactorParameters):
        self.params = params
        self.neutronics_params = NeutronicsParameters()
        self.results = {}

    def calculate_keff(self, burnup: float) -> float:
        """
        Calculate k-effective at given burnup.

        Simplified model - decreases with burnup.
        """
        enrichment = self.params.enrichment

        # Start at 1.0 + slight excess
        keff = 1.02 - 0.003 * burnup

        # Clamp to reality
        keff = max(0.9, min(1.03, keff))

        return keff

    def calculate_flux_distribution(self) -> np.ndarray:
        """
        Calculate radial flux distribution.

        Simplified cosine shape.
        """
        r = self.params.active_radius
        n_points = 20

        radii = np.linspace(0, r, n_points)
        flux = np.cos(1.5 * np.pi * radii / r)

        # Normalize
        flux = flux / np.max(flux) * self.neutronics_params.neutron_flux

        return flux

    def calculate_power_distribution(self) -> np.ndarray:
        """Calculate power distribution."""
        flux = self.calculate_flux_distribution()
        power = flux * self.params.power / np.sum(flux)

        return power

    def get_neutronics_state(self) -> Dict[str, float]:
        """Get current neutronics state."""
        return {
            "k_effective": self.calculate_keff(self.neutronics_params.burnup),
            "neutron_flux": self.neutronics_params.neutron_flux,
            "power_density": self.neutronics_params.core_power_density,
            "burnup": self.neutronics_params.burnup,
        }


class BurnupModel:
    """Fuel burnup calculations."""

    def __init__(self, params: ReactorParameters):
        self.params = params

    def calculate_isotopic_composition(self, burnup: float) -> Dict[str, float]:
        """Calculate isotopic composition at given burnup."""
        # Simplified - based on ORIGEN
        return {
            "u235": 35.0 * (1 - 0.5 * burnup / 50),  # kg initial
            "u238": 965.0,
            "pu239": 8.0 * (burnup / 50),
            "pu240": 2.5 * (burnup / 50),
            "fission_products": 30.0 * (burnup / 50),
        }

    def calculate_reactivity(self, burnup: float) -> float:
        """Calculate reactivity as function of burnup."""
        # Initially positive, eventually negative (requires burnable poison)
        reactivity = 0.03 - 0.0005 * burnup
        return max(-0.01, reactivity)

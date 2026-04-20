"""
Thermal Hydraulic Models
======================
Heat transfer and coolant flow in reactors.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class ThermalHydraulicParameters:
    """Thermal hydraulic parameters."""

    coolant_inlet_temp: float = 565.0  # K (PWR)
    coolant_outlet_temp: float = 615.0  # K
    coolant_flow: float = 70.0  # Mt/s (megatonnes/s, converted)
    core_power: float = 3000.0  # MW
    active_height: float = 4.0  # m
    active_radius: float = 1.8  # m
    fuel_rod_diameter: float = 0.008  # m


class ThermalHydraulicModel:
    """Thermal hydraulic calculations."""

    def __init__(self, params: ThermalHydraulicParameters):
        self.params = params

    def calculate_heat_transfer(self) -> Dict[str, float]:
        """Calculate heat transfer parameters."""
        p = self.params

        # Temperature rise
        delta_T = p.coolant_outlet_temp - p.coolant_inlet_temp

        # Power density
        core_volume = np.pi * p.active_radius**2 * p.active_height
        power_density = p.core_power / core_volume  # MW/m3 = 1000 kW/L

        # Cladding surface heat flux
        n_rods = 50000  # Typical PWR
        rod_surface = np.pi * p.fuel_rod_diameter * p.active_height * n_rods
        heat_flux = p.core_power * 1e6 / rod_surface  # W/m2

        return {
            "delta_T": delta_T,
            "power_density": power_density,
            "heat_flux": heat_flux,
            "clad_temp": p.coolant_outlet_temp + 200,  # Approximate
        }

    def calculate_DNBR(self) -> float:
        """
        Departure from Nucleate Boiling Ratio.

        Critical safety parameter - should be > 1.5
        """
        ht = self.calculate_heat_transfer()

        # Simplified - assuming adequate margin
        dnb_ratio = 2.0 - 0.3 * (ht["heat_flux"] / 100000)

        return max(1.0, dnb_ratio)

    def calculate_coolant_properties(self) -> Dict[str, float]:
        """Calculate coolant properties at operating conditions."""
        p = self.params

        # Simplified water properties at ~600K
        return {
            "density": 700.0,  # kg/m3
            "specific_heat": 5200.0,  # J/kg.K
            "thermal_conductivity": 0.5,  # W/m.K
            "viscosity": 0.0001,  # Pa.s
        }


class FuelPerformanceModel:
    """Fuel rod performance."""

    def __init__(self):
        self.params = None

    def calculate_fission_gas_release(self, burnup: float) -> float:
        """Calculate fission gas release to void."""
        # Simplified model
        release_fraction = 0.01 * (burnup / 50)
        return min(0.3, release_fraction)

    def calculate_fuel_swelling(self, burnup: float) -> float:
        """Calculate fuel swelling."""
        swelling = 0.01 * (burnup / 50)
        return swelling

    def calculate_cladding_stress(self, burnup: float) -> Dict[str, float]:
        """Calculate cladding stress state."""
        return {
            "hoop_stress": 100.0 + 5.0 * burnup,  # MPa
            "axial_stress": 20.0 + 1.0 * burnup,
            "equivalent_stress": 105.0 + 5.0 * burnup,
        }

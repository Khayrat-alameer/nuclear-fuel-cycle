"""
Reactor Operation Module - In-Reactor Fuel Behavior

This module provides models for nuclear reactor operation including:
- Fuel assembly and Core models
- Burnup calculations
- Neutronics (simplified)
- Fuel cycle length calculations
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FuelAssembly:
    """
    Model for a fuel assembly.

    Attributes:
        initial_enrichment: Initial U-235 enrichment (weight fraction)
        mass_kg: Heavy metal mass in kg U
        burnup_discharge: Discharge burnup in GWd/tU
        cycles_in_core: Number of cycles the assembly has seen
    """

    initial_enrichment: float = 0.045
    mass_kg: float = 200.0
    burnup_discharge: float = 50.0
    cycles_in_core: int = 1

    @property
    def initial_u235_kg(self) -> float:
        """Initial U-235 mass in kg."""
        return self.mass_kg * self.initial_enrichment

    @property
    def final_burnup(self) -> float:
        """Final burnup in MWd/kg."""
        return self.burnup_discharge / 1000

    def summary(self) -> dict:
        return {
            "enrichment_pct": self.initial_enrichment * 100,
            "mass_kg": self.mass_kg,
            "initial_U235_kg": self.initial_u235_kg,
            "discharge_burnup_gwd_t": self.burnup_discharge,
        }


@dataclass
class ReactorModel:
    """
    Model for a nuclear reactor.

    Attributes:
        name: Reactor name
        power_mwe: Net electrical power in MWe
        thermal_efficiency: Thermal to electrical efficiency
        cycle_length_days: Fuel cycle length in days
        refuel_days: Refueling downtime in days
        assemblies_in_core: Number of fuel assemblies in core
        assembly_mass_kg: Mass per fuel assembly in kg
    """

    name: str = "PWR"
    power_mwe: float = 1000.0
    thermal_efficiency: float = 0.33
    cycle_length_days: int = 18
    refuel_days: int = 30
    assemblies_in_core: int = 193
    assembly_mass_kg: float = 200.0

    @property
    def thermal_power_mwt(self) -> float:
        """Reactor thermal power in MWt."""
        return self.power_mwe / self.thermal_efficiency

    @property
    def annual_energy_mwh(self) -> float:
        """Annual electricity production in MWh."""
        total_days = self.cycle_length_days + self.refuel_days
        cycles_per_year = 365 / total_days
        return self.power_mwe * self.cycle_length_days * cycles_per_year

    @property
    def annual_production_gwh(self) -> float:
        """Annual electricity production in GWh."""
        return self.annual_energy_mwh / 1000

    @property
    def core_heavy_metal_t(self) -> float:
        """Core heavy metal loading in tonnes."""
        return (self.assemblies_in_core * self.assembly_mass_kg) / 1000

    @property
    def batch_reload_kg(self) -> float:
        """Annual fresh fuel requirement in kg."""
        total_days = self.cycle_length_days + self.refuel_days
        batches_per_year = 365 / total_days
        return self.assemblies_in_core * self.assembly_mass_kg * batches_per_year

    def fuel_requirement(
        self,
        burnup_gwd_t: float = 50,
        enrichment_pct: float = 4.5,
        tails_pct: float = 0.25,
    ) -> dict:
        """
        Calculate fuel cycle requirements.

        Parameters:
            burnup_gwd_t: Target discharge burnup in GWd/tU
            enrichment_pct: Fresh fuel enrichment in weight percent
            tails_pct: Tails (depleted) enrichment in weight percent

        Returns:
            Dictionary with fuel requirements
        """
        from .enrichment import EnrichmentCascade

        annual_hm = self.batch_reload_kg / 1000

        cascade = EnrichmentCascade(
            feed_mass=annual_hm,
            product_enrichment=enrichment_pct / 100,
            tails_enrichment=tails_pct / 100,
        )

        return {
            "fresh_fuel_kg": cascade.product_mass,
            "swu_required": cascade.swu_total,
            "depleted_u_kg": cascade.tails_mass,
            "feed_u_kg": annual_hm,
            "burnup_gwd_t": burnup_gwd_t,
            "enrichment_pct": enrichment_pct,
        }

    def energy_per_kg_u(self, burnup_gwd_t: float = 50) -> float:
        """Energy extracted per kg of uranium."""
        return burnup_gwd_t * 1000

    def summary(self) -> dict:
        return {
            "name": self.name,
            "power_mwe": self.power_mwe,
            "thermal_power_mwt": self.thermal_power_mwt,
            "core_hm_t": self.core_heavy_metal_t,
            "annual_reload_kg": self.batch_reload_kg,
            "annual_gwh": self.annual_production_gwh,
        }


class FuelBurnup:
    """
    Fuel burnup calculation model.
    """

    def __init__(
        self,
        initial_enrichment: float = 0.045,
        power_density_kw_kg: float = 40,
        batch_fraction: float = 0.25,
    ):
        self.initial_enrichment = initial_enrichment
        self.power_density_kw_kg = power_density_kw_kg
        self.batch_fraction = batch_fraction

    def calculate_burnup(self, cycle_days: int, mass_kg: float) -> float:
        """Calculate burnup after cycle."""
        burnup_mwd = self.power_density_kw_kg * cycle_days / mass_kg
        return burnup_mwd / 1000

    def equilibrium_composition(self, discharge_burnup_gwd_t: float = 50) -> dict:
        """
        Estimate equilibrium fuel composition at discharge.

        This is a simplified model.
        """
        u235_fraction = self.initial_enrichment * (
            1 - discharge_burnup_t_gwd(discharge_burnup_gwd_t) * 0.15
        )
        u238_fraction = 1 - u235_fraction

        pu_initial = self.initial_enrichment * 0.01
        pu_end = pu_initial + discharge_burnup_gwd_t * 0.0008

        return {
            "U235_fraction": max(0, u235_fraction),
            "U238_fraction": u238_fraction,
            "Pu_total_fraction": pu_end,
            "fission_products": discharge_burnup_gwd_t * 0.0001,
            " actinides": 1 - u235_fraction - u238_fraction - pu_end,
        }


def discharge_burnup_t_gwd(burnup_gwd_t: float) -> float:
    """Convert GWd/tU to MWd/kg."""
    return burnup_gwd_t * 1000


def calculate_reactor_cycle(
    power_mwe: float,
    burnup_gwd_t: float,
    enrichment_pct: float,
    capacity_factor: float = 0.90,
) -> dict:
    """
    Calculate key reactor fuel cycle parameters.

    Returns:
        Dictionary with fuel cycle requirements
    """
    energy_annual = power_mwe * 8760 * capacity_factor
    heavy_metal_annual = energy_annual / (burnup_gwd_t * 1e6)

    return {
        "annual_electricity_gwh": energy_annual / 1000,
        "annual_heavy_metal_t": heavy_metal_annual,
        "enrichment_pct": enrichment_pct,
    }


def calculate_batch_size(
    core_assemblies: int,
    assembly_mass_kg: float,
    refueling_interval_days: int,
    total_days: int,
) -> float:
    """
    Calculate the batch reload size.
    """
    batches_per_year = total_days / refueling_interval_days
    return core_assemblies * assembly_mass_kg / batches_per_year


if __name__ == "__main__":
    reactor = ReactorModel(
        name="典型PWR", power_mwe=1000, assemblies_in_core=193, assembly_mass_kg=200
    )

    print("=== Reactor Model ===")
    for k, v in reactor.summary().items():
        print(f"{k}: {v}")

    print("\n=== Fuel Requirements ===")
    req = reactor.fuel_requirement(burnup_gwd_t=50, enrichment_pct=4.5)
    for k, v in req.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

"""
Waste Disposal Module - Nuclear Waste Management

This module provides models for nuclear waste disposal including:
- Spent fuel interim storage
- Geological repository disposal
- High-level waste classification
- Waste package design
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WasteStream:
    """
    Model for a waste stream.

    Attributes:
        name: Waste stream name
        mass_kg: Mass in kg
        classification: HLW (high-level), ILW (intermediate), LLW (low-level)
        heat_generation_w_kg: Decay heat in W/kg
        activity_bq: Total activity in Becquerels
    """

    name: str = "Spent Fuel"
    mass_kg: float = 0.0
    classification: str = "HLW"
    heat_generation_w_kg: float = 0.0
    activity_bq: float = 0.0

    def cooling_time_years(self, target_heat_w_kg: float) -> float:
        """
        Calculate required cooling time to reach target heat load.

        Simple exponential decay model.
        """
        if self.heat_generation_w_kg <= target_heat_w_kg:
            return 0

        decay_constant = 0.035
        return -math.log(target_heat_w_kg / self.heat_generation_w_kg) / decay_constant

    def summary(self) -> dict:
        return {
            "name": self.name,
            "mass_kg": self.mass_kg,
            "classification": self.classification,
            "heat_w_kg": self.heat_generation_w_kg,
            "activity_Bq": self.activity_bq,
        }


import math


@dataclass
class InterimStorage:
    """
    Interim spent fuel storage model.

    Attributes:
        name: Storage facility name
        capacity_tHM: Storage capacity in tonnes heavy metal
        initial_heat_load_w_kg: Initial decay heat
    """

    name: str = "ISFSI"
    capacity_tHM: float = 10000.0
    initial_heat_load_w_kg: float = 2.0

    def store_fuel(self, spent_fuel_kg: float, storage_years: int) -> dict:
        """
        Calculate fuel storage conditions.

        Parameters:
            spent_fuel_kg: Mass of spent fuel
            storage_years: Years in storage

        Returns:
            Storage metrics
        """
        decay_factor = math.exp(-0.035 * storage_years)
        final_heat = self.initial_heat_load_w_kg * decay_factor

        return {
            "initial_heat_w_kg": self.initial_heat_load_w_kg,
            "storage_years": storage_years,
            "final_heat_w_kg": final_heat,
            "heat_reduction_pct": (1 - decay_factor) * 100,
            "capacity_remaining_tHM": self.capacity_tHM - spent_fuel_kg / 1000,
        }

    def summary(self) -> dict:
        return {
            "name": self.name,
            "capacity_tHM": self.capacity_tHM,
            "initial_heat_w_kg": self.initial_heat_load_w_kg,
        }


@dataclass
class GeologicalRepository:
    """
    Geological repository model for final disposal.
    """

    name: str = "Repository"
    capacity_tHM: float = 100000.0
    footprint_km2: float = 5.0
    construction_cost_M: float = 10000
    operation_cost_per_package: float = 500000

    def dispose_waste(
        self, waste_packages: int, waste_per_package_kg: float = 10000
    ) -> dict:
        """
        Calculate disposal requirements and costs.
        """
        total_waste = waste_packages * waste_per_package_kg

        return {
            "waste_packages": waste_packages,
            "total_waste_kg": total_waste,
            "capacity_used_tHM": total_waste / 1000,
            "capacity_remaining_tHM": self.capacity_tHM - total_waste / 1000,
            "construction_cost_M": self.construction_cost_M,
            "operation_cost_M": waste_packages * self.operation_cost_per_package / 1e6,
            "total_cost_M": self.construction_cost_M
            + waste_packages * self.operation_cost_per_package / 1e6,
            "cost_per_kg": (
                self.construction_cost_M * 1e6
                + waste_packages * self.operation_cost_per_package
            )
            / total_waste,
        }

    def package_heat_limit(self, repository_thermal_limit_w_kg: float = 0.5) -> int:
        """
        Calculate maximum packages per year based on thermal limit.
        """
        packages_per_year = 250
        return packages_per_year


@dataclass
class WasteFromSpentFuel:
    """
    Model for waste generation from spent fuel.
    """

    @staticmethod
    def calculate_waste(
        spent_fuel_kg: float, reprocessed: bool = False, burnup_gwd_t: float = 50
    ) -> dict:
        """
        Calculate waste from spent fuel.

        Parameters:
            spent_fuel_kg: Mass of spent fuel
            reprocessed: Whether fuel is reprocessed
            burnup_gwd_t: Discharge burnup

        Returns:
            Waste stream breakdown
        """
        if not reprocessed:
            return {
                "spent_fuel_kg": spent_fuel_kg,
                "hlw_kg": spent_fuel_kg,
                "ilw_kg": 0,
                "llw_kg": spent_fuel_kg * 0.01,
                "classification": "HLW",
            }

        return {
            "spent_fuel_kg": spent_fuel_kg,
            "hlw_kg": spent_fuel_kg * 0.02,
            "ilw_kg": spent_fuel_kg * 0.03,
            "llw_kg": spent_fuel_kg * 0.05,
            "classification": "Mixed",
        }


if __name__ == "__main__":
    storage = InterimStorage()
    print("=== Interim Storage ===")
    print(storage.summary())

    result = storage.store_fuel(100000, 40)
    print("\n=== After 40 Years Storage ===")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")

    repo = GeologicalRepository()
    print("\n=== Repository Disposal ===")
    repo_result = repo.dispose_waste(waste_packages=100)
    for k, v in repo_result.items():
        print(f"{k}: {v:.2f}")

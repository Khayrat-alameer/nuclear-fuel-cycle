"""
Reprocessing Module - Spent Fuel Reprocessing

This module provides models for spent nuclear fuel reprocessing including:
- PUREX process modeling
- Uranium and Plutonium recovery
- Minor actinide partitioning
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReprocessingPlant:
    """
    Model for a reprocessing plant (PUREX process).

    Attributes:
        name: Plant name
        capacity_tHM_per_year: Annual processing capacity in tonnes heavy metal
        u_recovery_rate: Uranium recovery efficiency
        pu_recovery_rate: Plutonium recovery efficiency
        minor_actinide_recovery: Minor actinide recovery fraction
    """

    name: str = "Reprocessing Plant"
    capacity_tHM_per_year: float = 800.0
    u_recovery_rate: float = 0.995
    pu_recovery_rate: float = 0.998
    minor_actinide_recovery: float = 0.95

    def process_spent_fuel(self, spent_fuel_kg: float) -> dict:
        """
        Process spent fuel and calculate product streams.

        Parameters:
            spent_fuel_kg: Mass of spent fuel in kg

        Returns:
            Dictionary with product streams
        """
        hm_mass = spent_fuel_kg

        u_recovered = hm_mass * 0.95 * self.u_recovery_rate
        pu_recovered = hm_mass * 0.01 * self.pu_recovery_rate
        ma_recovered = hm_mass * 0.005 * self.minor_actinide_recovery
        fission_products = hm_mass * 0.04

        return {
            "input_spent_fuel_kg": spent_fuel_kg,
            "recovered_uranium_kg": u_recovered,
            "recovered_plutonium_kg": pu_recovered,
            "recovered_actinides_kg": ma_recovered,
            "fission_products_kg": fission_products,
            "recovery_efficiency_pct": (u_recovered + pu_recovered) / hm_mass * 100,
        }

    def calculate_reprocessing_cost(
        self,
        spent_fuel_kg: float,
        capital_cost_M: float = 5000,
        operating_cost_per_kg: float = 1500,
        discount_rate: float = 0.08,
        plant_life_years: int = 40,
    ) -> dict:
        """Calculate reprocessing costs."""
        annual_capacity_kg = self.capacity_tHM_per_year * 1000
        years_operating = spent_fuel_kg / annual_capacity_kg

        annual_capital = (
            capital_cost_M
            * 1e6
            / ((1 - (1 + discount_rate) ** -plant_life_years) / discount_rate)
        )

        total_operating = operating_cost_per_kg * spent_fuel_kg

        return {
            "capital_cost_USD": annual_capital * years_operating,
            "operating_cost_USD": total_operating,
            "total_cost_USD": annual_capital * years_operating + total_operating,
            "cost_per_kg": (annual_capital * years_operating + total_operating)
            / spent_fuel_kg,
        }

    @property
    def annual_capacity_kg(self) -> float:
        return self.capacity_tHM_per_year * 1000

    def summary(self) -> dict:
        return {
            "name": self.name,
            "capacity_tHM_yr": self.capacity_tHM_per_year,
            "U_recovery_pct": self.u_recovery_rate * 100,
            "Pu_recovery_pct": self.pu_recovery_rate * 100,
            "MA_recovery_pct": self.minor_actinide_recovery * 100,
        }


@dataclass
class MOXFabrication:
    """
    MOX (Mixed Oxide) fuel fabrication model.
    """

    capacity_tHM_per_year: float = 100.0
    fabrication_loss: float = 0.03
    pu_fabrication_cost_per_kg: float = 2000
    blending_loss: float = 0.01

    def fabricate_mox(
        self,
        plutonium_kg: float,
        depleted_u_kg: float,
        target_pu_fraction: float = 0.08,
    ) -> dict:
        """
        Fabricate MOX fuel.

        Parameters:
            plutonium_kg: Plutonium available in kg
            depleted_u_kg: Depleted uranium available in kg
            target_pu_fraction: Target Pu fraction in MOX

        Returns:
            MOX production metrics
        """
        mox_kg = (plutonium_kg / target_pu_fraction) * (1 - self.fabrication_loss)
        du_needed = mox_kg - plutonium_kg

        if du_needed > depleted_u_kg:
            mox_kg = (
                depleted_u_kg / (1 - target_pu_fraction) * (1 - self.fabrication_loss)
            )
            plutonium_kg = mox_kg * target_pu_fraction

        fabrication_cost = mox_kg * self.pu_fabrication_cost_per_kg

        return {
            "mox_fuel_kg": mox_kg,
            "plutonium_used_kg": plutonium_kg,
            "depleted_u_used_kg": du_needed,
            "fabrication_cost_USD": fabrication_cost,
            "cost_per_kg_MOX": fabrication_cost / mox_kg if mox_kg > 0 else 0,
        }


def calculate_pu_content(spent_fuel_kg: float, burnup_gwd_t: float) -> float:
    """Estimate Pu content in spent fuel."""
    return spent_fuel_kg * (0.0008 * burnup_gwd_t + 0.0005)


if __name__ == "__main__":
    plant = ReprocessingPlant()
    print("=== Reprocessing Plant ===")
    for k, v in plant.summary().items():
        print(f"{k}: {v}")

    result = plant.process_spent_fuel(10000)
    print("\n=== Processing 10,000 kg Spent Fuel ===")
    for k, v in result.items():
        print(f"{k}: {v:.2f}")

    mox = MOXFabrication()
    mox_result = mox.fabricate_mox(plutonium_kg=100, depleted_u_kg=1000)
    print("\n=== MOX Fabrication ===")
    for k, v in mox_result.items():
        print(f"{k}: {v:.2f}")

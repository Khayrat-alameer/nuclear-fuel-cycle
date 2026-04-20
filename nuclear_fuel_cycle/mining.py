"""
Mining Module - Uranium Mining and Extraction Models

This module provides models for uranium mining operations including:
- Resource estimation
- Extraction process modeling (in-situ recovery, heap leaching)
- Environmental impact assessment
- Economic analysis
"""

import math
from dataclasses import dataclass
from typing import Optional


U3O8_TO_U = 0.8480
U_TO_U3O8 = 1 / U3O8_TO_U
U235_MASS = 235.044
URANIUM_ATOMIC_MASS = 238.050


@dataclass
class UraniumResource:
    """
    Model for a uranium resource deposit.

    Attributes:
        name: Resource name
        reserves_tU: Recoverable reserves in tonnes of uranium
        grade_ppm: Average uranium grade in parts per million
        resource_tU: Total resource in tonnes uranium (if different from reserves)
        mining_method: Primary mining method (open_pit, underground, ISR)
        recovery_factor: Overall recovery factor
    """

    name: str = "Unknown"
    reserves_tU: float = 0.0
    grade_ppm: float = 0.0
    resource_tU: float = 0.0
    mining_method: str = "open_pit"
    recovery_factor: float = 0.9

    def __post_init__(self):
        if self.resource_tU is None:
            self.resource_tU = self.reserves_tU / self.recovery_factor
        if self.resource_tU is None:
            self.resource_tU = self.reserves_tU

    @property
    def total_resource_tU(self) -> float:
        """Total resource in tonnes U."""
        return self.resource_tU if self.resource_tU is not None else 0.0

    @property
    def ore_tonnage(self) -> float:
        """Required ore tonnage in tonnes."""
        if self.grade_ppm <= 0:
            return 0
        res = self.resource_tU if self.resource_tU is not None else 0.0
        return (res * 1e6) / self.grade_ppm

    @property
    def u3o8_reserves_kg(self) -> float:
        """Reserves in kg U3O8."""
        return self.reserves_tU * 1000 * U_TO_U3O8

    def summary(self) -> dict:
        return {
            "name": self.name,
            "reserves_tU": self.reserves_tU,
            "reserves_kg_U3O8": self.u3o8_reserves_kg,
            "grade_ppm": self.grade_ppm,
            "ore_tonnage_t": self.ore_tonnage,
            "mining_method": self.mining_method,
        }


@dataclass
class MiningOperation:
    """
    Model for a uranium mining operation.

    Attributes:
        name: Operation name
        ore_processed_tpd: Ore processed per day (tonnes/day)
        head_grade_ppm: Feed grade to mill
        recovery_rate: Overall recovery rate to yellowcake
        operating_days: Operating days per year
    """

    name: str = "Mine"
    ore_processed_tpd: float = 10000
    head_grade_ppm: float = 1000
    recovery_rate: float = 0.85
    operating_days: float = 350

    @property
    def annual_ore_t(self) -> float:
        """Annual ore processed in tonnes."""
        return self.ore_processed_tpd * self.operating_days

    @property
    def annual_u3o8_production_kg(self) -> float:
        """Annual U3O8 production in kg."""
        annual_u_kg = (self.annual_ore_t * self.head_grade_ppm) / 1e6
        return annual_u_kg * U_TO_U3O8 * self.recovery_rate

    @property
    def annual_uranium_kg(self) -> float:
        """Annual uranium production in kg."""
        return (self.annual_ore_t * self.head_grade_ppm) / 1e6 * self.recovery_rate

    def summary(self) -> dict:
        return {
            "name": self.name,
            "ore_processed_tpd": self.ore_processed_tpd,
            "head_grade_ppm": self.head_grade_ppm,
            "recovery_rate": self.recovery_rate,
            "annual_U3O8_kg": self.annual_u3o8_production_kg,
            "annual_U_kg": self.annual_uranium_kg,
        }


class InSituRecovery:
    """
    Model for In-Situ Recovery (ISR) uranium extraction.

    ISR is a solution mining method where groundwater with oxidants
    is injected to dissolve uranium in-place.
    """

    def __init__(
        self,
        wellfield_hectares: float = 100,
        average_grade_ppm: float = 800,
        recovery_rate: float = 0.75,
        solution_strength_gpl: float = 1.0,
    ):
        self.wellfield_hectares = wellfield_hectares
        self.average_grade_ppm = average_grade_ppm
        self.recovery_rate = recovery_rate
        self.solution_strength_gpl = solution_strength_gpl

    @property
    def in_place_u_t(self) -> float:
        """Uranium in place in tonnes."""
        ore_tonnage = self.wellfield_hectares * 10000 * 50
        return (ore_tonnage * self.average_grade_ppm) / 1e6

    @property
    def recoverable_u_t(self) -> float:
        """Recoverable uranium in tonnes."""
        return self.in_place_u_t * self.recovery_rate

    @property
    def recoverable_u3o8_t(self) -> float:
        """Recoverable U3O8 in tonnes."""
        return self.recoverable_u_t * U_TO_U3O8

    def summary(self) -> dict:
        return {
            "wellfield_hectares": self.wellfield_hectares,
            "average_grade_ppm": self.average_grade_ppm,
            "in_place_uranium_t": self.in_place_u_t,
            "recoverable_uranium_t": self.recoverable_u_t,
            "recoverable_U3O8_t": self.recoverable_u3o8_t,
        }


class MiningCostModel:
    """
    Economic model for uranium mining operations.
    """

    def __init__(
        self,
        capital_cost_M: float = 500,
        operating_cost_per_kg_U3O8: float = 100,
        production_rate_kg_U3O8_per_year: float = 2000000,
        discount_rate: float = 0.08,
        project_life_years: int = 20,
    ):
        self.capital_cost_M = capital_cost_M
        self.operating_cost_per_kg_U3O8 = operating_cost_per_kg_U3O8
        self.production_rate_kg_U3O8_per_year = production_rate_kg_U3O8_per_year
        self.discount_rate = discount_rate
        self.project_life_years = project_life_years

    @property
    def annual_operating_cost_M(self) -> float:
        """Annual operating cost in million $."""
        return (
            self.operating_cost_per_kg_U3O8 * self.production_rate_kg_U3O8_per_year
        ) / 1e6

    @property
    def unit_cost_per_kg(self) -> float:
        """Levelized cost in $/kg U3O8."""
        npv_capital = self.capital_cost_M
        annual_op = self.annual_operating_cost_M

        npv_op = annual_op * (
            (1 - (1 + self.discount_rate) ** -self.project_life_years)
            / self.discount_rate
        )

        total_npv = npv_capital + npv_op
        total_production = (
            self.production_rate_kg_U3O8_per_year * self.project_life_years
        )

        return (total_npv / total_production) * 1e6

    def production_cost_breakdown(
        self,
        electricity_cost_per_kwh: float = 0.05,
        acid_cost_per_tonne: float = 150,
        acid_kg_per_kg_U: float = 15,
        electricity_kwh_per_kg_U: float = 35,
        labor_cost_percent: float = 0.30,
        consumables_cost_percent: float = 0.25,
        maintenance_cost_percent: float = 0.15,
        overhead_cost_percent: float = 0.10,
    ) -> dict:
        """Break down of production costs."""
        direct_cost = (
            electricity_cost_per_kwh * electricity_kwh_per_kg_U
            + acid_cost_per_tonne * acid_kg_per_kg_U / 1000
        )

        return {
            "direct_production": direct_cost,
            "labor": direct_cost * labor_cost_percent,
            "consumables": direct_cost * consumables_cost_percent,
            "maintenance": direct_cost * maintenance_cost_percent,
            "overhead": direct_cost * overhead_cost_percent,
            "total_cost": direct_cost
            / (
                1
                - labor_cost_percent
                - consumables_cost_percent
                - maintenance_cost_percent
                - overhead_cost_percent
            ),
        }


def calculate_mine_life(reserves_tU: float, annual_production_tU: float) -> float:
    """Calculate mine life in years."""
    return reserves_tU / annual_production_tU


def grade_tonnage_relationship(
    cutoff_grade_ppm: float,
    total_resource_t: float,
    average_grade_ppm: float,
    dispersion_factor: float = 1.5,
) -> dict:
    """
    Calculate recoverable uranium at different cutoff grades.

    This uses a simple log-normal approximation for grade-tonnage curves.
    """
    if average_grade_ppm <= cutoff_grade_ppm:
        return {
            "cutoff_grade_ppm": cutoff_grade_ppm,
            "recoverable_tU": 0,
            "resource_t": 0,
        }

    tonnage_factor = math.exp(
        -((cutoff_grade_ppm / average_grade_ppm) ** (1 / dispersion_factor) - 1)
    )
    recoverable_t = total_resource_t * tonnage_factor

    return {
        "cutoff_grade_ppm": cutoff_grade_ppm,
        "resource_t": total_resource_t * (1 - tonnage_factor * 0.1),
        "recoverable_tU": max(0, recoverable_t),
    }


if __name__ == "__main__":
    resource = UraniumResource(
        name="Typical Sandstone Deposit",
        reserves_tU=50000,
        grade_ppm=1200,
        mining_method="ISR",
    )

    print("=== Uranium Resource ===")
    for k, v in resource.summary().items():
        print(f"{k}: {v}")

    print("\n=== Mining Operation ===")
    mine = MiningOperation(
        name="ISR Wellfield",
        ore_processed_tpd=5000,
        head_grade_ppm=800,
        recovery_rate=0.75,
    )
    for k, v in mine.summary().items():
        print(f"{k}: {v}")

    print("\n=== Cost Model ===")
    cost = MiningCostModel(
        capital_cost_M=200,
        operating_cost_per_kg_U3O8=80,
        production_rate_kg_U3O8_per_year=1000000,
    )
    print(f"Levelized cost: ${cost.unit_cost_per_kg:.2f}/kg U3O8")

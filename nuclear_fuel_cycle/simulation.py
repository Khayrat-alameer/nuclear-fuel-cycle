"""
Simulation Runner - End-to-End Fuel Cycle Simulation

This module provides a simulation framework that chains all nuclear fuel cycle
stages together for comprehensive analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .enrichment import EnrichmentCascade
from .reactor import ReactorModel
from .mining import MiningOperation
from .fabrication import FuelFabrication
from .reprocessing import ReprocessingPlant
from .waste_disposal import InterimStorage, GeologicalRepository


@dataclass
class FuelCycleSimulation:
    """
    Complete fuel cycle simulation model.

    This simulates a once-through fuel cycle with options for recycling.
    """

    reactor: ReactorModel = field(default_factory=ReactorModel)

    enrichment_enrichment_pct: float = 4.5
    enrichment_tails_pct: float = 0.25

    mining_grade_ppm: float = 1000
    mining_recovery: float = 0.85

    fabrication_loss: float = 0.02

    use_reprocessing: bool = False
    reprocessing_capacity_tHM: float = 0.0

    storage_years: int = 40

    def run_once_through(self) -> dict:
        """
        Run once-through fuel cycle simulation.

        Returns:
            Complete fuel cycle results
        """
        mining = MiningOperation(
            head_grade_ppm=self.mining_grade_ppm,
            recovery_rate=self.mining_recovery,
            ore_processed_tpd=10000,
        )

        enrichment = EnrichmentCascade(
            feed_mass=self.reactor.batch_reload_kg,
            product_enrichment=self.enrichment_enrichment_pct / 100,
            tails_enrichment=self.enrichment_tails_pct / 100,
        )

        fabrication = FuelFabrication(process_loss_rate=self.fabrication_loss)

        storage = InterimStorage(initial_heat_load_w_kg=2.0)

        storage_result = storage.store_fuel(
            spent_fuel_kg=self.reactor.batch_reload_kg, storage_years=self.storage_years
        )

        return {
            "mining": {
                "annual_U3O8_kg": mining.annual_u3o8_production_kg,
                "required_grade_ppm": self.mining_grade_ppm,
            },
            "enrichment": {
                "feed_kg": enrichment.feed_mass,
                "product_kg": enrichment.product_mass,
                "tails_kg": enrichment.tails_mass,
                "swu_required": enrichment.swu_total,
                "enrichment_pct": self.enrichment_enrichment_pct,
                "tails_pct": self.enrichment_tails_pct,
            },
            "fabrication": {
                "input_kg": enrichment.product_mass,
                "fabrication_yield": 1 - self.fabrication_loss,
                "output_kg": enrichment.product_mass * (1 - self.fabrication_loss),
            },
            "reactor": {
                "power_mwe": self.reactor.power_mwe,
                "annual_load_kg": self.reactor.batch_reload_kg,
                "annual_gwh": self.reactor.annual_production_gwh,
            },
            "storage": {
                "heat_after_40yr_w_kg": storage_result["final_heat_w_kg"],
            },
            "total_swu": enrichment.swu_total,
            "total_fresh_fuel_kg": enrichment.product_mass
            * (1 - self.fabrication_loss),
        }

    def run_with_recycling(self) -> dict:
        """
        Run fuel cycle with reprocessing and MOX recycling.

        Returns:
            Results including recycled material
        """
        base_result = self.run_once_through()

        if not self.use_reprocessing or self.reprocessing_capacity_tHM <= 0:
            return {**base_result, "recycling": {"status": "not_used"}}

        reprocessing = ReprocessingPlant(
            capacity_tHM_per_year=self.reprocessing_capacity_tHM
        )

        reproc_result = reprocessing.process_spent_fuel(
            spent_fuel_kg=self.reactor.batch_reload_kg
        )

        storage = InterimStorage()
        storage_result = storage.store_fuel(
            spent_fuel_kg=self.reactor.batch_reload_kg, storage_years=self.storage_years
        )

        return {
            **base_result,
            "reprocessing": {
                "U_recovered_kg": reproc_result["recovered_uranium_kg"],
                "Pu_recovered_kg": reproc_result["recovered_plutonium_kg"],
                "fp_kg": reproc_result["fission_products_kg"],
                "capacity_tHM": self.reprocessing_capacity_tHM,
            },
            "storage": {
                "heat_after_40yr_w_kg": storage_result["final_heat_w_kg"],
            },
        }

    def run(self) -> dict:
        """Run the appropriate fuel cycle based on configuration."""
        if self.use_reprocessing:
            return self.run_with_recycling()
        return self.run_once_through()

    def summary(self) -> dict:
        """Summary of simulation configuration."""
        return {
            "reactor_power_mwe": self.reactor.power_mwe,
            "enrichment_pct": self.enrichment_enrichment_pct,
            "target_burnup_gwd_t": 50,
            "use_reprocessing": self.use_reprocessing,
            "storage_years": self.storage_years,
        }


@dataclass
class MultiReactorSimulation:
    """
    Simulation for multiple reactors.
    """

    reactors: List[ReactorModel] = field(default_factory=list)
    use_reprocessing: bool = False
    reprocessing_capacity_tHM: float = 0.0

    def add_reactor(self, reactor: ReactorModel):
        """Add a reactor to the simulation."""
        self.reactors.append(reactor)

    def run(self) -> dict:
        """Run simulation for all reactors."""
        total_annual_gwh = 0
        total_fresh_fuel_kg = 0
        total_swu = 0
        total_spent_fuel_kg = 0

        results = []

        for reactor in self.reactors:
            sim = FuelCycleSimulation(
                reactor=reactor,
                use_reprocessing=self.use_reprocessing,
                reprocessing_capacity_tHM=self.reprocessing_capacity_tHM,
            )
            result = sim.run()
            results.append(result)

            total_annual_gwh += result.get("reactor", {}).get("annual_gwh", 0)
            total_fresh_fuel_kg += result.get("enrichment", {}).get("product_kg", 0)
            total_swu += result.get("enrichment", {}).get("swu_required", 0)
            total_spent_fuel_kg += result.get("reactor", {}).get("annual_load_kg", 0)

        return {
            "individual_results": results,
            "total": {
                "reactors": len(self.reactors),
                "annual_electricity_gwh": total_annual_gwh,
                "fresh_fuel_kg": total_fresh_fuel_kg,
                "total_swu": total_swu,
                "spent_fuel_kg": total_spent_fuel_kg,
            },
        }


def run_annual_simulation(
    reactor_power_mwe: float,
    enrichment_pct: float = 4.5,
    burnup_gwd_t: float = 50,
    use_reprocessing: bool = False,
    reprocessing_capacity_tHM: float = 0.0,
) -> dict:
    """
    Convenience function to run a simple annual simulation.

    Parameters:
        reactor_power_mwe: Reactor power
        enrichment_pct: Fuel enrichment
        burnup_gwd_t: Discharge burnup
        use_reprocessing: Enable reprocessing
        reprocessing_capacity_tHM: Reprocessing capacity

    Returns:
        Annual fuel cycle results
    """
    reactor = ReactorModel(power_mwe=reactor_power_mwe)

    sim = FuelCycleSimulation(
        reactor=reactor,
        enrichment_enrichment_pct=enrichment_pct,
        use_reprocessing=use_reprocessing,
        reprocessing_capacity_tHM=reprocessing_capacity_tHM,
    )

    return sim.run()


if __name__ == "__main__":
    print("=== Once-Through Fuel Cycle ===")
    print("1 x 1000 MWe PWR")

    sim = FuelCycleSimulation(
        reactor=ReactorModel(power_mwe=1000),
        enrichment_enrichment_pct=4.5,
        storage_years=40,
    )

    result = sim.run_once_through()

    print("\n--- Mining ---")
    print(f"U3O8 required: {result['mining']['annual_U3O8_kg']:.0f} kg")

    print("\n--- Enrichment ---")
    print(f"Product: {result['enrichment']['product_kg']:.0f} kg")
    print(f"SWU: {result['enrichment']['swu_required']:.0f}")

    print("\n--- Reactor ---")
    print(f"Annual GWh: {result['reactor']['annual_gwh']:.0f}")

    print("\n--- Storage ---")
    print(f"Heat after 40yr: {result['storage']['heat_after_40yr_w_kg']:.4f} W/kg")

    print("\n=== With Reprocessing ===")
    sim_reproc = FuelCycleSimulation(
        reactor=ReactorModel(power_mwe=1000),
        enrichment_enrichment_pct=4.5,
        use_reprocessing=True,
        reprocessing_capacity_tHM=800,
    )
    result_reproc = sim_reproc.run_with_reprocessing()
    print(f"U recovered: {result_reproc['reprocessing']['U_recovered_kg']:.0f} kg")
    print(f"Pu recovered: {result_reproc['reprocessing']['Pu_recovered_kg']:.0f} kg")

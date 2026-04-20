"""
Fabrication Module - Nuclear Fuel Fabrication

This module provides models for nuclear fuel fabrication including:
- UO2 fuel pellet manufacturing
- Fuel assembly fabrication
- Quality control and losses
"""

from dataclasses import dataclass


U_TO_UO2 = 1.0 / 0.8815


@dataclass
class FuelFabrication:
    """
    Model for fuel fabrication facility.

    Attributes:
        name: Facility name
        capacity_tU_per_year: Annual fabrication capacity in tU
        process_loss_rate: Fraction lost during fabrication
        enrichment_services: External enrichment service cost
    """

    name: str = "Fuel Fab Plant"
    capacity_tU_per_year: float = 2000.0
    process_loss_rate: float = 0.02
    enrichment_services_cost_per_swu: float = 130.0

    @property
    def uo2_conversion_factor(self) -> float:
        """Kg UO2 per kg U."""
        return U_TO_UO2

    def fabrice_uo2_pellets(self, uranium_kg: float) -> dict:
        """
        Calculate UO2 pellet production from uranium.

        Parameters:
            uranium_kg: Input uranium mass in kg

        Returns:
            Dictionary with production metrics
        """
        uo2_produced = uranium_kg * self.process_yield * U_TO_UO2
        process_loss = uranium_kg * self.process_loss_rate

        return {
            "input_uranium_kg": uranium_kg,
            "uo2_pellets_kg": uo2_produced,
            "process_loss_kg": process_loss,
            "yield_pct": (1 - self.process_loss_rate) * 100,
        }

    @property
    def process_yield(self) -> float:
        """Process yield (1 - loss rate)."""
        return 1 - self.process_loss_rate

    @property
    def annual_capacity_kg(self) -> float:
        """Annual capacity in kg."""
        return self.capitality_tU_per_year * 1000

    def fabrication_cost(
        self,
        uranium_kg: float,
        enrichment_pct: float,
        swu_required: float,
        conversion_cost_per_kg: float = 250,
        fabrication_cost_per_kg: float = 300,
    ) -> dict:
        """
        Calculate total fuel fabrication cost.

        Parameters:
            uranium_kg: Uranium input in kg
            enrichment_pct: Fuel enrichment in weight percent
            swu_required: SWU required for enrichment
            conversion_cost_per_kg: Conversion cost $/kg U
            fabrication_cost_per_kg: Fabrication cost $/kg UO2

        Returns:
            Cost breakdown
        """
        conversion_cost = uranium_kg * conversion_cost_per_kg
        enrich_cost = swu_required * self.enrichment_services_cost_per_swu

        uo2_output = uranium_kg * self.process_yield * U_TO_UO2
        fab_cost = uo2_output * fabrication_cost_per_kg

        return {
            "conversion_cost_USD": conversion_cost,
            "enrichment_cost_USD": enrich_cost,
            "fabrication_cost_USD": fab_cost,
            "total_cost_USD": conversion_cost + enrich_cost + fab_cost,
            "cost_per_kg_UO2": (conversion_cost + enrich_cost + fab_cost) / uo2_output,
        }

    def summary(self) -> dict:
        return {
            "name": self.name,
            "capacity_tU_yr": self.capacity_tU_per_year,
            "process_loss_pct": self.process_loss_rate * 100,
            "swu_cost_per_swu": self.enrichment_services_cost_per_swu,
        }


@dataclass
class FuelAssemblyFab:
    """
    Fuel assembly fabrication model.
    """

    assemblies_per_year: int = 500
    rods_per_assembly: int = 264
    pellet_loading_kg: float = 200.0
    fabrication_loss_rate: float = 0.01

    @property
    def total_pellets_kg(self) -> float:
        """Total pellets per year."""
        return self.assemblies_per_year * self.pellet_loading_kg

    def pellet_to_assembly_cost(
        self, pellet_cost_per_kg: float = 300, assembly_cost_per: float = 50000
    ) -> dict:
        """Assembly fabrication cost breakdown."""
        pellet_cost = self.total_pellets_kg * pellet_cost_per_kg
        assembly_cost = self.assemblies_per_year * assembly_cost_per

        return {
            "pellet_cost_USD": pellet_cost,
            "assembly_cost_USD": assembly_cost,
            "total_cost_USD": pellet_cost + assembly_cost,
            "cost_per_assembly_USD": (pellet_cost + assembly_cost)
            / self.assemblies_per_year,
        }


def uo2_from_uranium(uranium_kg: float, conversion_rate: float = 0.98) -> float:
    """Convert uranium to UO2."""
    return uranium_kg * conversion_rate * U_TO_UO2


if __name__ == "__main__":
    fab = FuelFabrication(name="Commercial Plant", capacity_tU_per_year=1200)
    print("=== Fuel Fabrication ===")
    for k, v in fab.summary().items():
        print(f"{k}: {v}")

    result = fab.fabrication_cost(
        uranium_kg=1000, enrichment_pct=4.5, swu_required=5000
    )
    print("\n=== Cost Breakdown (1000 kg U) ===")
    for k, v in result.items():
        print(f"{k}: ${v:,.0f}")

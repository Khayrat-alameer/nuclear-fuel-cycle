#!/usr/bin/env python3
"""
Fabrication Simulation Runner

Simple simulation that creates results matching the data parameters.
"""

import json
import math

U_TO_UO2 = 1.0 / 0.8815


def run_simulation():
    """Run fuel fabrication simulation."""

    uranium_kg = 31838
    enrichment_pct = 4.5
    swu_required = 218765

    conversion_cost = uranium_kg * 250
    enrichment_cost = swu_required * 130
    uo2_produced = uranium_kg * U_TO_UO2 * 0.98
    fabrication_cost = uo2_produced * 300
    total_cost = conversion_cost + enrichment_cost + fabrication_cost

    assemblies = 172
    pellets_per_assembly = 264
    pellet_kg = 200

    assembly_result = {
        "assemblies": assemblies,
        "rods_per_assembly": pellets_per_assembly,
        "pellet_cost": pellets_per_assembly * pellet_kg * 300,
        "labor_cost": assemblies * 50000,
    }

    sim_result = {
        "simulation_type": "fuel_fabrication",
        "simulation_date": "2025-04-20",
        "facility": {
            "name": "Commercial Fuel Fabrication Plant",
            "capacity_tU_yr": 1200,
            "process_loss_pct": 2.0,
        },
        "pellet_production": {
            "uranium_input_kg": uranium_kg,
            "uo2_pellets_kg": round(uo2_produced, 2),
            "conversion_cost_USD": round(conversion_cost, 2),
            "enrichment_cost_USD": round(enrichment_cost, 2),
            "fabrication_cost_USD": round(fabrication_cost, 2),
            "total_cost_USD": round(total_cost, 2),
            "cost_per_kg_UO2": round(total_cost / uo2_produced, 2),
        },
        "assembly_production": assembly_result,
    }

    return sim_result


if __name__ == "__main__":
    result = run_simulation()
    print("=== Fuel Fabrication Simulation ===")
    print(f"UO2 Pellets: {result['pellet_production']['uo2_pellets_kg']:.0f} kg")
    print(f"Total Cost: ${result['pellet_production']['total_cost_USD']:,.0f}")
    print(f"Cost/kg UO2: ${result['pellet_production']['cost_per_kg_UO2']:.2f}")

    output_file = "C:/Users/khayrat/Desktop/hermes-agent/nuclear-fuel-cycle/fabrication/simulations/fabrication_simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

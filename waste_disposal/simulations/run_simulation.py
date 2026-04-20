#!/usr/bin/env python3
"""
Waste Disposal Simulation Runner

Simulates interim storage and geological disposal.
"""

import json
import math


def run_simulation():
    """Run waste disposal simulation."""

    spent_fuel_kg = 29350
    storage_capacity_tHM = 10000
    repository_capacity_tHM = 100000

    initial_heat = 2.0
    decay_constant = 0.035

    temp_conditions = [5, 10, 20, 40, 100]
    heat_by_age = {}
    for years in temp_conditions:
        heat_by_age[str(years)] = round(
            initial_heat * math.exp(-decay_constant * years), 4
        )

    repository_heat_limit = 0.5
    storage_needed_years = (
        -math.log(repository_heat_limit / initial_heat) / decay_constant
    )

    packages = math.ceil(spent_fuel_kg / 10000)
    repository_construction = 10000
    repository_operation = packages * 500000

    total_cost = repository_construction + repository_operation
    cost_per_kg = total_cost / spent_fuel_kg

    sim_result = {
        "simulation_type": "waste_disposal",
        "simulation_date": "2025-04-20",
        "interim_storage": {
            "capacity_tHM": storage_capacity_tHM,
            "initial_heat_W_kg": initial_heat,
            "decay_constant_yr": decay_constant,
            "cooling_time_to_0.5W_kg": round(storage_needed_years, 1),
            "heat_by_age_years": heat_by_age,
        },
        "disposal": {
            "repository_capacity_tHM": repository_capacity_tHM,
            "repository_depth_m": 500,
            "geologic_formation": "Clay",
            "waste_packages": packages,
            "construction_cost_M USD": repository_construction,
            "operation_cost_M USD": round(repository_operation / 1e6, 2),
            "total_cost_M USD": round(total_cost / 1e6, 2),
            "cost_per_kg": round(cost_per_kg, 2),
        },
        "packages": {
            "canister_material": "Carbon Steel",
            "canister_thickness_mm": 100,
            "buffer_material": "Bentonite",
            "thermal_limit_W_kg": repository_heat_limit,
            "glass_form": "Borosilicate",
        },
    }

    return sim_result


if __name__ == "__main__":
    result = run_simulation()
    print("=== Waste Disposal Simulation ===")
    print(
        f"Storage Cooling Time: {result['interim_storage']['cooling_time_to_0.5W_kg']} years"
    )
    print(f"Repository Packages: {result['disposal']['waste_packages']}")
    print(f"Total Cost: ${result['disposal']['total_cost_M USD']:.1f}M")
    print(f"Cost/kg: ${result['disposal']['cost_per_kg']:.2f}")

    output_file = "C:/Users/khayrat/Desktop/hermes-agent/nuclear-fuel-cycle/waste_disposal/simulations/waste_disposal_simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

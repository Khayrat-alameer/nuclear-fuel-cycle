#!/usr/bin/env python3
"""
Reactor Operation Simulation Runner

Simulates PWR reactor fuel cycle operations.
"""

import json
import math


def run_simulation():
    """Run reactor operation simulation."""

    power_mwe = 1000
    power_mwt = power_mwe / 0.33
    thermal_efficiency = 0.33
    capacity_factor = 0.90

    annual_energy = power_mwe * 8760 * capacity_factor
    cycle_days = 18
    refuel_days = 30
    total_days = cycle_days + refuel_days

    cycles_per_year = 365 / total_days

    assemblies = 193
    assembly_mass = 200
    core_hm = assemblies * assembly_mass / 1000

    batch_kg = assemblies * assembly_mass * cycles_per_year

    burnup_gwd_t = 50
    energy_per_kg = burnup_gwd_t * 1000

    annual_fuel_kg = annual_energy / energy_per_kg

    u235_initial = 0.045
    u235_discharge = u235_initial * (1 - burnup_gwd_t * 0.003)

    sim_result = {
        "simulation_type": "reactor_operation",
        "simulation_date": "2025-04-20",
        "reactor": {
            "name": "典型PWR",
            "power_mwe": power_mwe,
            "power_mwt": round(power_mwt, 2),
            "thermal_efficiency": thermal_efficiency,
            "capacity_factor": capacity_factor,
            "cycle_days": cycle_days,
            "refuel_days": refuel_days,
        },
        "core": {
            "assemblies": assemblies,
            "assembly_mass_kg": assembly_mass,
            "core_heavy_metal_t": core_hm,
            "active_fuel_length_m": 3.6,
            "rods_per_assembly": 264,
        },
        "fuel_cycle": {
            "batch_reload_kg": round(batch_kg, 2),
            "cycles_per_year": round(cycles_per_year, 2),
            "annual_heavy_metal_t": round(annual_fuel_kg / 1000, 2),
            "burnup_gwd_t": burnup_gwd_t,
            "initial_enrichment_pct": u235_initial * 100,
            "discharge_enrichment_pct": round(u235_discharge * 100, 3),
        },
        "performance": {
            "annual_electricity_gwh": round(annual_energy / 1000, 2),
            "capacity_factor": capacity_factor * 100,
            "peak_linear_power_w_cm": 180,
            "avg_linear_power_w_cm": 120,
        },
    }

    return sim_result


if __name__ == "__main__":
    result = run_simulation()
    print("=== Reactor Operation Simulation ===")
    print(f"Power: {result['reactor']['power_mwe']} MWe")
    print(f"Annual Electricity: {result['performance']['annual_electricity_gwh']} GWh")
    print(f"Burnup: {result['fuel_cycle']['burnup_gwd_t']} GWd/tU")
    print(f"Batch Reload: {result['fuel_cycle']['batch_reload_kg']:.0f} kg")

    output_file = "C:/Users/khayrat/Desktop/hermes-agent/nuclear-fuel-cycle/reactor_operation/simulations/reactor_simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

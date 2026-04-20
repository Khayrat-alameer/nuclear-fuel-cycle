#!/usr/bin/env python3
"""
Reprocessing Simulation Runner

Simulates PUREX spent fuel reprocessing.
"""

import json
import math


def run_simulation():
    """Run reprocessing simulation."""

    spent_fuel_kg = 29350
    burnup_gwd_t = 50
    cooling_years = 5

    u_recovery = 0.995
    pu_recovery = 0.998
    ma_recovery = 0.90

    u_content = 0.95
    pu_content = 0.008 + burnup_gwd_t * 0.0001
    ma_content = 0.003
    fp_content = 1 - u_content - pu_content - ma_content

    u_recovered = spent_fuel_kg * u_content * u_recovery
    pu_recovered = spent_fuel_kg * pu_content * pu_recovery
    ma_recovered = spent_fuel_kg * ma_content * ma_recovery
    fp_recovered = spent_fuel_kg * fp_content

    capacity_tHM = 800
    capital_cost = 5000
    operating_cost_per_kg = 1500

    capital_annual = capital_cost * 1e6 / 30
    operating_total = operating_cost_per_kg * spent_fuel_kg / 1000

    waste_hlw = spent_fuel_kg * 0.02
    waste_ilw = spent_fuel_kg * 0.03
    waste_llw = spent_fuel_kg * 0.05

    sim_result = {
        "simulation_type": "reprocessing",
        "simulation_date": "2025-04-20",
        "feed": {
            "spent_fuel_kg": spent_fuel_kg,
            "burnup_gwd_t": burnup_gwd_t,
            "cooling_years": cooling_years,
        },
        "products": {
            "uranium_recovered_kg": round(u_recovered, 2),
            "plutonium_recovered_kg": round(pu_recovered, 2),
            "minor_actinides_kg": round(ma_recovered, 2),
            "fission_products_kg": round(fp_recovered, 2),
        },
        "recovery": {
            "uranium_efficiency_pct": u_recovery * 100,
            "plutonium_efficiency_pct": pu_recovery * 100,
            "actinide_efficiency_pct": ma_recovery * 100,
        },
        "economics": {
            "capacity_tHM_yr": capacity_tHM,
            "capital_cost_M USD": capital_cost,
            "annual_capital_M USD": round(capital_annual / 1e6, 2),
            "operating_cost_M USD": round(operating_total / 1e6, 2),
            "cost_per_kg": round(operating_total / spent_fuel_kg, 2),
        },
        "waste": {
            "hlw_kg": round(waste_hlw, 2),
            "ilw_kg": round(waste_ilw, 2),
            "llw_kg": round(waste_llw, 2),
        },
    }

    return sim_result


if __name__ == "__main__":
    result = run_simulation()
    print("=== Reprocessing Simulation ===")
    print(f"Spent Fuel: {result['feed']['spent_fuel_kg']:.0f} kg")
    print(f"U Recovered: {result['products']['uranium_recovered_kg']:.0f} kg")
    print(f"Pu Recovered: {result['products']['plutonium_recovered_kg']:.1f} kg")
    print(f"Cost/kg: ${result['economics']['cost_per_kg']:.2f}")

    output_file = "C:/Users/khayrat/Desktop/hermes-agent/nuclear-fuel-cycle/reprocessing/simulations/reprocessing_simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

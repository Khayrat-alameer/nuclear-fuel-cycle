"""
Streamlit Web App - Reactor Operation Simulator
=========================================
Nuclear reactor neutronics and thermal hydraulics.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reactor_operation.models.neutronics import (
    NeutronicsModel,
    NeutronicsParameters,
    BurnupModel,
    ReactorParameters,
)
from reactor_operation.models.thermal_hydraulic import (
    ThermalHydraulicModel,
    ThermalHydraulicParameters,
    FuelPerformanceModel,
)

st.set_page_config(page_title="Reactor Operation", page_icon="🔥", layout="wide")

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("Reactor Parameters")

sim_option = st.sidebar.selectbox(
    "Simulation", ["Neutronics", "Thermal Hydraulics", "Both"], index=2
)

st.sidebar.markdown("---")
st.sidebar.header("Core Parameters")

power = st.sidebar.slider(
    "Thermal Power (MW)", min_value=500.0, max_value=5000.0, value=3000.0
)

enrichment = st.sidebar.slider(
    "Fuel Enrichment (%)", min_value=2.0, max_value=20.0, value=3.5
)

active_height = st.sidebar.slider(
    "Active Height (m)", min_value=2.0, max_value=6.0, value=4.0
)

st.sidebar.markdown("---")
st.sidebar.header("Operating Parameters")

inlet_temp = st.sidebar.slider(
    "Coolant Inlet Temp (K)", min_value=500.0, max_value=600.0, value=565.0
)

outlet_temp = st.sidebar.slider(
    "Coolant Outlet Temp (K)", min_value=550.0, max_value=700.0, value=615.0
)

burnup = st.sidebar.slider(
    "Fuel Burnup (MWd/kgU)", min_value=0.0, max_value=200.0, value=50.0
)

# ============================================================================
# Main Content
# ============================================================================

st.title("🔥 Reactor Operation Simulator")
st.markdown("### Neutronics & Thermal Hydraulics")

# ============================================================================
# Neutronics
# ============================================================================

if sim_option in ["Neutronics", "Both"]:
    st.markdown("## 1. Neutronics")

    @st.cache_data
    def run_neutronics(power, enrichment, height, burnup):
        params = ReactorParameters(
            power=power, enrichment=enrichment / 100, active_height=height
        )

        neutronics = NeutronicsModel(params)
        burnup_model = BurnupModel(params)

        keff = neutronics.calculate_keff(burnup)
        flux = np.max(neutronics.calculate_flux_distribution())
        iso = burnup_model.calculate_isotopic_composition(burnup)
        reactivity = burnup_model.calculate_reactivity(burnup)

        return keff, flux, iso, reactivity

    with st.spinner("Calculating neutronics..."):
        keff, flux, iso, reactivity = run_neutronics(
            power, enrichment, active_height, burnup
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("k-effective", f"{keff:.4f}")

        with col2:
            st.metric("Max Flux (n/cm²/s)", f"{flux:.2e}")

        with col3:
            st.metric("Reactivity", f"{reactivity:+.3f}")

        with col4:
            st.metric("U-235 (kg)", f"{iso['u235']:.1f}")

        st.markdown("---")

# ============================================================================
# Thermal Hydraulics
# ============================================================================

if sim_option in ["Thermal Hydraulics", "Both"]:
    st.markdown("## 2. Thermal Hydraulics")

    @st.cache_data
    def run_thermal(power, height, inlet, outlet, burnup):
        params = ThermalHydraulicParameters(
            core_power=power,
            active_height=height,
            coolant_inlet_temp=inlet,
            coolant_outlet_temp=outlet,
        )

        th = ThermalHydraulicModel(params)
        fp = FuelPerformanceModel()

        ht = th.calculate_heat_transfer()
        dnb = th.calculate_DNBR()
        fgr = fp.calculate_fission_gas_release(burnup)
        stress = fp.calculate_cladding_stress(burnup)

        return ht, dnb, fgr, stress

    with st.spinner("Calculating thermal..."):
        ht, dnb, fgr, stress = run_thermal(
            power, active_height, inlet_temp, outlet_temp, burnup
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Delta T (K)", f"{ht['delta_T']:.0f}")

        with col2:
            st.metric("DNBR", f"{dnb:.2f}")

        with col3:
            st.metric("FGR (%)", f"{fgr * 100:.1f}")

        with col4:
            st.metric("Clad Stress (MPa)", f"{stress['hoop_stress']:.0f}")

        st.markdown("---")

# ============================================================================
# Info
# ============================================================================

with st.expander("ℹ️ reactor Parameters"):
    st.markdown("""
    **PWR Operating Conditions:**
    - Inlet: ~565 K (292°C)
    - Outlet: ~615 K (342°C)
    - DNBR > 1.5 required
    - Max cladding stress: < 200 MPa
    """)

st.caption("Nuclear Fuel Cycle | Reactor Operation")

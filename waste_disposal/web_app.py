"""
Streamlit Web App - Waste Disposal Simulator
=====================================
Geological repository performance.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from waste_disposal.models.repository import (
    RepositoryModel,
    RepositoryParameters,
    RepositoryPerformanceModel,
)
from waste_disposal.models.radionuclide_transport import (
    RadionuclideTransportModel,
    TransportParameters,
    EngineeredBarrierModel,
)

st.set_page_config(page_title="Waste Disposal", page_icon="🏚️", layout="wide")

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("Repository Parameters")

sim_option = st.sidebar.selectbox(
    "Simulation", ["Repository", "Transport", "Both"], index=2
)

st.sidebar.markdown("---")
st.sidebar.header("Repository Design")

depth = st.sidebar.slider("Depth (m)", min_value=200.0, max_value=1000.0, value=500.0)

host_rock = st.sidebar.selectbox("Host Rock", ["clay", "granite", "salt"], index=0)

can_material = st.sidebar.selectbox(
    "Canister Material", ["steel", "copper", "titanium"], index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Transport Parameters")

nuclide = st.sidebar.selectbox(
    "Key Radionuclide", ["cs137", "sr90", "u238", "pu239", "am241"], index=3
)

distance = st.sidebar.slider(
    "Migration Distance (m)", min_value=10.0, max_value=1000.0, value=100.0
)

# ============================================================================
# Main Content
# ============================================================================

st.title("🏚️ Nuclear Waste Disposal")
st.markdown("### Repository Performance & Transport")

# ============================================================================
# Repository
# ============================================================================

if sim_option in ["Repository", "Both"]:
    st.markdown("## 1. Repository Performance")

    @st.cache_data
    def run_repository(depth, rock, can_mat):
        params = RepositoryParameters(
            repository_depth=depth, host_rock=rock, can_material=can_mat
        )

        model = RepositoryModel(params)
        perf = RepositoryPerformanceModel()

        containment = model.calculate_containment_time()
        heat = model.calculate_heat_generation()
        risk = perf.calculate_risk()

        return containment, heat, risk

    with st.spinner("Analyzing repository..."):
        containment, heat, risk = run_repository(depth, host_rock, can_material)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Containment (kyr)", f"{containment['canister']:.0f}")

        with col2:
            st.metric("Initial Heat (W)", f"{heat['initial']:.0f}")

        with col3:
            st.metric("After 1k yr (W)", f"{heat['after_1000_years']:.0f}")

        with col4:
            st.metric("Risk", f"{risk:.2e}")

        st.markdown("---")

# ============================================================================
# Transport
# ============================================================================

if sim_option in ["Transport", "Both"]:
    st.markdown("## 2. Radionuclide Transport")

    @st.cache_data
    def run_transport(nuclide, distance):
        params = TransportParameters()

        model = RadionuclideTransportModel(params)
        barrier = EngineeredBarrierModel()

        R = model.calculate_retardation_factor(nuclide)
        t_migration = model.calculate_migration_time(nuclide, distance)
        dose = model.calculate_annual_dose(nuclide)
        barrier_perf = barrier.get_barrier_performance()

        return R, t_migration, dose, barrier_perf

    with st.spinner("Calculating transport..."):
        R, t_mig, dose, barrier_perf = run_transport(nuclide, distance)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Retardation Factor", f"{R:.0f}")

        with col2:
            st.metric(f"Migration Time (yr)", f"{t_mig:.0f}")

        with col3:
            st.metric("Dose (mSv/yr)", f"{dose:.2e}")

        with col4:
            st.metric("Barrier Integrity", f"{barrier_perf['canister_integrity']:.0%}")

        st.markdown("---")

# ============================================================================
# Info
# ============================================================================

with st.expander("ℹ️ Repository Types"):
    st.markdown("""
    **Host Rock Options:**
    - **Clay**: Low permeability, good containment
    - **Granite**: Stable, proven in Sweden/Finland
    - **Salt**: Self-healing, USA/Germany experience
    
    **Safety Case:** Multiple barriers provide containment for 100,000+ years
    """)

st.caption("Nuclear Fuel Cycle | Waste Disposal")

"""
Streamlit Web App - Spent Fuel Reprocessing Simulator
============================================
PUREX process simulation.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reprocessing.models.purex_process import (
    PUREXModel,
    PUREXParameters,
    RecoveryEfficiencyModel,
)
from reprocessing.models.separation_chemistry import (
    SeparationChemistryModel,
    SolventExtractionParams,
)
from reprocessing.models.waste_stream import WasteStreamModel, WasteParameters

st.set_page_config(page_title="Spent Fuel Reprocessing", page_icon="🔄", layout="wide")

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("Process Parameters")

process_option = st.sidebar.selectbox(
    "Process", ["PUREX Extraction", "Waste Analysis", "Both"], index=2
)

st.sidebar.markdown("---")
st.sidebar.header("PUREX Parameters")

feed_flow = st.sidebar.slider(
    "Feed Flow Rate (L/h)", min_value=10.0, max_value=500.0, value=100.0
)

acid_conc = st.sidebar.slider(
    "Aqueous HNO3 (M)", min_value=0.5, max_value=4.0, value=2.0
)

extraction_stages = st.sidebar.slider(
    "Extraction Stages", min_value=4, max_value=20, value=12
)

tbp_conc = st.sidebar.slider(
    "TBP Concentration", min_value=0.1, max_value=0.4, value=0.30
)

st.sidebar.markdown("---")
st.sidebar.header("Fuel Parameters")

burnup = st.sidebar.slider(
    "Fuel Burnup (GWd/tU)", min_value=10.0, max_value=200.0, value=50.0
)

cooling = st.sidebar.slider(
    "Cooling Time (years)", min_value=1.0, max_value=40.0, value=5.0
)

recovery = st.sidebar.slider(
    "Recovery Efficiency (%)", min_value=90.0, max_value=99.9, value=99.0
)

# ============================================================================
# Main Content
# ============================================================================

st.title("🔄 Spent Fuel Reprocessing")
st.markdown("### PUREX Process & Waste Characterization")

# ============================================================================
# PUREX Extraction
# ============================================================================

if process_option in ["PUREX Extraction", "Both"]:
    st.markdown("## 1. PUREX Extraction")

    @st.cache_data
    def run_purex(feed_flow, acid_conc, extraction_stages, tbp_conc):
        params = PUREXParameters(
            feed_flow_rate=feed_flow,
            aqueous_acid_conc=acid_conc,
            extraction_stages=extraction_stages,
        )

        model = PUREXModel(params)
        results = model.simulate_extraction()
        eff = model.get_recovery_efficiency()

        return results, eff

    with st.spinner("Simulating extraction..."):
        purex_results, purex_eff = run_purex(
            feed_flow, acid_conc, extraction_stages, tbp_conc
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("U Recovery", f"{purex_eff['u_recovery']:.1%}")

        with col2:
            st.metric("Product Assay", f"{purex_eff['product_purity']:.2f}")

        with col3:
            st.metric("Stages", str(purex_results["stages"]))

        with col4:
            st.metric("Loss to Tailings", f"{purex_eff['tailings_loss']:.2%}")

        st.markdown("---")

# ============================================================================
# Waste Analysis
# ============================================================================

if process_option in ["Waste Analysis", "Both"]:
    st.markdown("## 2. Waste Analysis")

    @st.cache_data
    def run_waste(burnup, cooling, recovery):
        params = WasteParameters(
            burnup=burnup, cooling_time=cooling, recovery_efficiency=recovery / 100
        )

        model = WasteStreamModel(params)
        fp = model.calculate_fission_products()
        act = model.calculate_actinides()
        vol = model.get_waste_volume()
        classification = model.get_waste_classification()

        return fp, act, vol, classification

    with st.spinner("Analyzing waste..."):
        fp, act, vol, classification = run_waste(burnup, cooling, recovery)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Cs-137 (PBq/t)", f"{fp['cs137']:.1f}")

        with col2:
            st.metric("Sr-90 (PBq/t)", f"{fp['sr90']:.1f}")

        with col3:
            st.metric("HLW Volume (L/t)", f"{vol['hlw_liters']:.0f}")

        with col4:
            st.metric("Classification", classification[:3])

        st.markdown("---")

# ============================================================================
# Info
# ============================================================================

with st.expander("ℹ️ PUREX Process"):
    st.markdown("""
    **PUREX**: Plutonium Uranium Reduction Extraction
    
    - Extracts U and Pu from spent fuel using TBP solvent
    - Extract: 30% TBP in dodecane
    - Typical recovery: >99% U, >99.5% Pu
    - Waste: HLW from fission products
    """)

st.caption("Nuclear Fuel Cycle | Reprocessing Module")

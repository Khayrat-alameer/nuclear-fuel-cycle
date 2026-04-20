"""
Streamlit Web App - Uranium Enrichment Cascade Simulator
====================================================
Interactive demonstration of gas centrifuge cascade modeling.
"""

import streamlit as st
import numpy as np
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import local modules
from enrichment.models.cascade_model import CentrifugeCascadeModel, CascadeParameters

st.set_page_config(
    page_title="Uranium Enrichment Simulator", page_icon="☢️", layout="wide"
)

# ============================================================================
# Sidebar - Configuration
# ============================================================================

st.sidebar.header("⚙️ Cascade Parameters")

feed_assay = st.sidebar.slider(
    "Feed Assay (U-235)",
    min_value=0.001,
    max_value=0.20,
    value=0.00711,
    format="%.4f",
    help="U-235 concentration in feed uranium. Natural uranium is ~0.00711 (0.711%)",
)

product_assay = st.sidebar.slider(
    "Product Assay (U-235)",
    min_value=0.01,
    max_value=0.95,
    value=0.035,
    format="%.3f",
    help="Target enrichment level. ~3-5% = reactor grade, ~90% = weapons grade",
)

tails_assay = st.sidebar.slider(
    "Tails Assay",
    min_value=0.0001,
    max_value=0.005,
    value=0.0025,
    format="%.4f",
    help="Depleted uranium assay (waste stream)",
)

feed_flow_rate = st.sidebar.slider(
    "Feed Flow Rate (kg/h)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0
)

st.sidebar.markdown("---")

st.sidebar.header("🔧 Cascade Settings")

stages = st.sidebar.slider("Number of Stages", min_value=5, max_value=50, value=15)

machines = st.sidebar.slider(
    "Number of Centrifuges", min_value=100, max_value=5000, value=1500, step=100
)

separation_factor = st.sidebar.slider(
    "Separation Factor (α)",
    min_value=1.05,
    max_value=1.50,
    value=1.20,
    step=0.01,
    help="Single stage separation factor",
)

simulation_time = st.sidebar.slider(
    "Simulation Time (hours)", min_value=1.0, max_value=72.0, value=24.0
)

# ============================================================================
# Main Content
# ============================================================================

st.title("☢️ Uranium Enrichment Cascade Simulator")
st.markdown("### Gas Centrifuge Cascade Modeling & Simulation")

# Validate inputs
if tails_assay >= feed_assay:
    st.error("❌ Tails assay must be less than feed assay")
    st.stop()

if product_assay <= feed_assay:
    st.error("❌ Product assay must be greater than feed assay")
    st.stop()


# Run simulation
@st.cache_data
def run_simulation(
    feed_assay,
    product_assay,
    tails_assay,
    feed_flow_rate,
    stages,
    machines,
    separation_factor,
    simulation_time,
):
    params = CascadeParameters(
        feed_assay=feed_assay,
        feed_flow_rate=feed_flow_rate,
        product_assay=product_assay,
        tails_assay=tails_assay,
        separation_factor=separation_factor,
        machine_count=machines,
        stages=stages,
        time_step=1.0,
        simulation_time=simulation_time,
    )

    model = CentrifugeCascadeModel(params)
    results = model.simulate_dynamic()
    performance = model.get_cascade_performance()

    return model, results, performance


# Run the simulation
with st.spinner("Running cascade simulation..."):
    model, results, performance = run_simulation(
        feed_assay,
        product_assay,
        tails_assay,
        feed_flow_rate,
        stages,
        machines,
        separation_factor,
        simulation_time,
    )

# ============================================================================
# Key Metrics Dashboard
# ============================================================================

st.markdown("### 📊 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Product Assay",
        f"{performance['product_assay_actual']:.4f}",
        f"{(performance['product_assay_actual'] - product_assay) * 100:.2f}% vs target",
    )

with col2:
    st.metric(
        "Tails Assay",
        f"{performance['tails_assay_actual']:.4f}",
        f"{(tails_assay - performance['tails_assay_actual']) * 100:.2f}% vs target",
    )

with col3:
    st.metric("Separation Efficiency", f"{performance['separation_efficiency']:.1%}")

with col4:
    st.metric(
        "Total SWU",
        f"{performance['total_swu']:.1f}",
        help="Separative Work Units - measure of enrichment work done",
    )

st.markdown("---")

# ============================================================================
# Visualizations
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Stage Profile", "📉 Dynamic Evolution", "📋 Summary"])

with tab1:
    st.markdown("#### Stage-wise Assay Profile")

    stage_assays_final = results["stage_assays"][-1, :]
    stage_indices = np.arange(len(stage_assays_final))

    chart_data = {
        "Stage": stage_indices,
        "U-235 Assay": stage_assays_final,
        "Target Profile": [
            model._ideal_stage_assay(i) for i in range(len(stage_assays_final))
        ],
    }

    st.line_chart(chart_data, x="Stage", y=["U-235 Assay", "Target Profile"])

    st.caption(f"Optimal feed stage: {performance['feed_stage']}")

with tab2:
    st.markdown("#### Assay Evolution Over Time")

    time_data = []
    for t in range(0, len(results["time"]), 5):
        if t < len(results["stage_assays"]):
            time_data.append(
                {
                    "Time (h)": results["time"][t],
                    "Top Stage": results["stage_assays"][t, -1],
                    "Bottom Stage": results["stage_assays"][t, 0],
                }
            )

    time_chart_data = {
        i: [row[i] for row in time_data]
        for i in ["Time (h)", "Top Stage", "Bottom Stage"]
    }
    st.line_chart(time_chart_data, x="Time (h)", y=["Top Stage", "Bottom Stage"])

with tab3:
    st.markdown("#### Summary")

    summary_data = {
        "Parameter": [
            "Feed Assay",
            "Product Assay (Target)",
            "Product Assay (Actual)",
            "Tails Assay (Target)",
            "Tails Assay (Actual)",
            "Feed Flow Rate",
            "Centrifuges",
            "Stages",
            "Separation Factor",
            "Total SWU",
            "SWU per kg Feed",
            "Feed Stage",
        ],
        "Value": [
            f"{feed_assay:.4f}",
            f"{product_assay:.4f}",
            f"{performance['product_assay_actual']:.4f}",
            f"{tails_assay:.4f}",
            f"{performance['tails_assay_actual']:.4f}",
            f"{feed_flow_rate:.1f} kg/h",
            str(machines),
            str(stages),
            f"{separation_factor:.2f}",
            f"{performance['total_swu']:.2f}",
            f"{performance['swu_per_kg_feed']:.4f}",
            str(performance["feed_stage"]),
        ],
    }

    st.table(summary_data)

# ============================================================================
# Information Box
# ============================================================================

st.markdown("---")

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("ℹ️ About SWU"):
        st.markdown("""
        **Separative Work Unit (SWU)** is the standard measure of enrichment work.
        
        - 1 SWU = One kg of uranium enriched from 0.0% to 100% U-235
        - Typical reactor fuel (3-5%): ~200-300 SWU per kg enriched
        - Weapons grade (90%+): ~2,000+ SWU per kg enriched
        """)

with info_col2:
    with st.expander("ℹ️ Enrichment Levels"):
        st.markdown("""
        - **Natural Uranium**: ~0.711% U-235
        - **LEU (Low Enriched)**: <5% U-235 (reactor fuel)
        - **HEU (Highly Enriched)**: >20% U-235
        - **Weapon Grade**: >90% U-235
        """)

st.markdown("---")
st.caption(
    "Nuclear Fuel Cycle Project | Enrichment Module | Based on scientific models"
)

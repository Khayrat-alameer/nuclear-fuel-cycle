"""
Streamlit Web App - Uranium Mining Simulator
=========================================
Interactive demonstration of uranium mining simulation.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mining.models.resource_estimation import (
    ResourceEstimationModel,
    ResourceEstimationParameters,
    DrillholeData,
)
from mining.models.extraction_efficiency import (
    ExtractionEfficiencyModel,
    HydrogeologicalParameters,
    LeachingParameters,
)


def generate_sample_drillhole_data(n_holes=50, x_range=1000, y_range=1000):
    """Generate synthetic drillhole data for demonstration with realistic ore zone."""
    np.random.seed(42)
    coords = np.random.uniform([0, 0, -200], [x_range, y_range, 0], (n_holes, 3))

    # Create ore zone in center - higher grades in middle
    grades = []
    for coord in coords:
        x, y, z = coord
        cx, cy = x_range / 2, y_range / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (x_range / 2)
        base_grade = np.random.lognormal(-3, 1)  # Background low grade
        if dist < 0.5:  # In ore zone
            zone_grade = np.random.uniform(0.05, 0.5)  # Higher grades
            grade = zone_grade * (1 - dist) + base_grade * dist
        else:
            grade = base_grade
        grades.append(grade)

    grades = np.array(grades)
    grades = np.clip(grades, 0.001, 2.0)
    depths = np.random.uniform(50, 200, n_holes)
    return DrillholeData(coordinates=coords, grades=grades, depths=depths)


st.set_page_config(page_title="Uranium Mining Simulator", page_icon="⛏️", layout="wide")

# ============================================================================
# Sidebar - Configuration
# ============================================================================

st.sidebar.header("Mining Parameters")

simulation_type = st.sidebar.selectbox(
    "Simulation Type", ["Resource Estimation", "Extraction", "Both"], index=2
)

st.sidebar.markdown("---")
st.sidebar.header("Resource Estimation")

grid_resolution = st.sidebar.slider(
    "Grid Resolution (m)", min_value=25.0, max_value=200.0, value=50.0
)

grade_cutoff = st.sidebar.slider(
    "Grade Cutoff (% U3O8)", min_value=0.01, max_value=0.20, value=0.05, step=0.01
)

x_range = st.sidebar.slider(
    "X Range (m)", min_value=100.0, max_value=2000.0, value=1000.0
)

y_range = st.sidebar.slider(
    "Y Range (m)", min_value=100.0, max_value=2000.0, value=1000.0
)

st.sidebar.markdown("---")
st.sidebar.header("Extraction Parameters")

leachant_type = st.sidebar.selectbox("Leachant Type", ["acid", "alkaline"], index=0)

initial_pH = st.sidebar.slider(
    "Initial pH", min_value=0.5, max_value=4.0, value=1.5, step=0.1
)

injection_rate = st.sidebar.slider(
    "Injection Rate (m³/s)",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f",
)

porosity = st.sidebar.slider(
    "Porosity", min_value=0.1, max_value=0.4, value=0.25, step=0.05
)

simulation_time_days = st.sidebar.slider(
    "Simulation Time (days)", min_value=30, max_value=730, value=365
)

# ============================================================================
# Main Content
# ============================================================================

st.title("⛏️ Uranium Mining Simulator")
st.markdown("### Resource Estimation & Extraction Modeling")

# ============================================================================
# Resource Estimation
# ============================================================================

if simulation_type in ["Resource Estimation", "Both"]:
    st.markdown("## 1. Resource Estimation")

    @st.cache_data
    def run_resource_estimation(grid_resolution, grade_cutoff, x_range, y_range):
        params = ResourceEstimationParameters(
            grid_resolution=grid_resolution,
            grade_cutoff=grade_cutoff,
            x_range=(0.0, x_range),
            y_range=(0.0, y_range),
            method="kriging",
        )

        model = ResourceEstimationModel(params)
        # Generate sample drillhole data for demo
        drillhole_data = generate_sample_drillhole_data(
            n_holes=50, x_range=x_range, y_range=y_range
        )
        model.set_drillhole_data(drillhole_data)
        results = model.estimate_resources()

        return results

    with st.spinner("Running resource estimation..."):
        try:
            resource_results = run_resource_estimation(
                grid_resolution, grade_cutoff, x_range, y_range
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Resource (kg)",
                    f"{resource_results.get('total_resource_kg', 0):,.0f}",
                )

            with col2:
                st.metric(
                    "Average Grade (% U3O8)",
                    f"{resource_results.get('average_grade', 0):.3f}",
                )

            with col3:
                st.metric("Cutoff Grade (% U3O8)", f"{grade_cutoff:.2f}")

            with col4:
                st.metric(
                    "Recoverable (kg)",
                    f"{resource_results.get('recoverable_kg', 0):,.0f}",
                )

            st.markdown("---")

        except Exception as e:
            st.warning(f"Resource estimation: {str(e)[:100]}...")

# ============================================================================
# Extraction Efficiency
# ============================================================================

if simulation_type in ["Extraction", "Both"]:
    st.markdown("## 2. Extraction Efficiency")

    @st.cache_data
    def run_extraction(
        leachant_type, initial_pH, injection_rate, porosity, simulation_time_days
    ):
        hydro_params = HydrogeologicalParameters(porosity=porosity)

        leach_params = LeachingParameters(
            leachant_type=leachant_type,
            initial_pH=initial_pH,
            injection_rate=injection_rate,
            simulation_time=simulation_time_days * 86400,
        )

        model = ExtractionEfficiencyModel(hydro_params, leach_params)
        # Run reactive transport simulation
        sim_results = model.simulate_reactive_transport(
            domain_size=(100, 100, 50), grid_resolution=5.0
        )
        results = model.calculate_extraction_efficiency()

        return results

    with st.spinner("Running extraction simulation..."):
        try:
            extraction_results = run_extraction(
                leachant_type,
                initial_pH,
                injection_rate,
                porosity,
                simulation_time_days,
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Extraction Efficiency",
                    f"{extraction_results.get('extraction_efficiency', 0):.1%}",
                )

            with col2:
                st.metric(
                    "Final Recovery (kg)",
                    f"{extraction_results.get('final_recovery_kg', 0):,.1f}",
                )

            with col3:
                st.metric(
                    "Avg Recovery Rate (kg/s)",
                    f"{extraction_results.get('average_recovery_rate_kg_per_s', 0):.4f}",
                )

            with col4:
                st.metric(
                    "Simulation Time",
                    f"{extraction_results.get('simulation_time_s', 0) / 86400:.0f} days",
                )

            st.markdown("---")

        except Exception as e:
            st.warning(f"Extraction: {str(e)[:100]}...")

# ============================================================================
# Summary
# ============================================================================

with st.expander("📋 Simulation Summary"):
    st.markdown("### Current Parameters")

    summary_data = {
        "Parameter": [
            "Simulation Type",
            "Grid Resolution",
            "Grade Cutoff",
            "X Range",
            "Y Range",
            "Leachant Type",
            "Initial pH",
            "Injection Rate",
            "Porosity",
            "Simulation Time",
        ],
        "Value": [
            simulation_type,
            f"{grid_resolution} m",
            f"{grade_cutoff:.2f}%",
            f"{x_range} m",
            f"{y_range} m",
            leachant_type,
            f"{initial_pH:.1f}",
            f"{injection_rate:.4f} m³/s",
            f"{porosity:.2f}",
            f"{simulation_time_days} days",
        ],
    }

    st.table(summary_data)

# ============================================================================
# Information
# ============================================================================

st.markdown("---")

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("ℹ️ About Uranium Mining"):
        st.markdown("""
        **Uranium Mining Methods:**
        - **Open Pit**: Surface mining for shallow deposits
        - **Underground**: For deeper orebodies
        - **In-Situ Leaching (ISL)**: Solution mining underground
        - **Heap Leaching**: Surface solution processing
        
        **Key Metrics:**
        - Grade: % U3O8 in ore
        - Recovery Rate: % of uranium extracted
        - Cutoff Grade: Minimum economic grade
        """)

with info_col2:
    with st.expander("ℹ️ Grade Classification"):
        st.markdown("""
        **Uranium Ore Grades:**
        - **High Grade**: > 1% U3O8
        - **Medium Grade**: 0.1 - 1% U3O8
        - **Low Grade**: 0.01 - 0.1% U3O8
        - **Very Low Grade**: < 0.01% U3O8
        
        **Typical Recovery:**
        - ISL: 70-90%
        - Heap Leaching: 60-80%
        - Conventional: 85-95%
        """)

st.markdown("---")
st.caption("Nuclear Fuel Cycle Project | Mining Module | Based on scientific models")

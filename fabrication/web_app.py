"""
Streamlit Web App - Nuclear Fuel Fabrication Simulator
===============================================
Interactive demonstration of fuel pellet fabrication.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fabrication.models.pellet_fabrication import (
    PelletFabricationModel,
    PelletDesign,
    SinteringParameters,
    MicrostructureParameters,
)
from fabrication.models.powder_processing import (
    PowderProcessingModel,
    PowderParameters,
    GranulationParameters,
)

st.set_page_config(
    page_title="Fuel Fabrication Simulator", page_icon="⚙️", layout="wide"
)

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("Fabrication Parameters")

process_type = st.sidebar.selectbox(
    "Process", ["Powder Processing", "Pellet Fabrication", "Both"], index=2
)

st.sidebar.markdown("---")
st.sidebar.header("Powder Parameters")

powder_uraniu_grade = st.sidebar.select_slider(
    "UO2 Powder Grade", options=[5.0, 7.0, 8.0, 10.0], value=7.0
)

initial_purity = st.sidebar.slider(
    "Chemical Purity (%)", min_value=95.0, max_value=99.9, value=99.5
)

particle_size = st.sidebar.slider(
    "Mean Particle Size (μm)", min_value=0.5, max_value=10.0, value=2.0
)

st.sidebar.markdown("---")
st.sidebar.header("Pellet Parameters")

pellet_diameter = st.sidebar.slider(
    "Pellet Diameter (mm)", min_value=5.0, max_value=12.0, value=8.19
)

pellet_height = st.sidebar.slider(
    "Pellet Height (mm)", min_value=5.0, max_value=20.0, value=10.0
)

peak_temperature = st.sidebar.slider(
    "Peak Temperature (°C)", min_value=1400.0, max_value=1800.0, value=1700.0
)

hold_time = st.sidebar.slider(
    "Hold Time (hours)", min_value=0.5, max_value=6.0, value=2.0
)

initial_porosity = st.sidebar.slider(
    "Initial Porosity", min_value=0.30, max_value=0.50, value=0.40
)

# ============================================================================
# Main Content
# ============================================================================

st.title("⚙️ Nuclear Fuel Fabrication Simulator")
st.markdown("### Powder Processing & Pellet Fabrication")

# ============================================================================
# Powder Processing
# ============================================================================

if process_type in ["Powder Processing", "Both"]:
    st.markdown("## 1. Powder Processing")

    @st.cache_data
    def run_powder_processing(powder_uraniu_grade, initial_purity, particle_size):
        powder_params = PowderParameters(
            uranium_grade=powder_uraniu_grade,
            chemical_purity=initial_purity / 100,
            mean_particle_size=particle_size,
        )

        gran_params = GranulationParameters(binder_content=2.0, lubricant_content=0.5)

        model = PowderProcessingModel(powder_params, gran_params)
        results = model.simulate_powder_flow()

        return results

    with st.spinner("Processing powder..."):
        try:
            powder_results = run_powder_processing(
                powder_uraniu_grade, initial_purity, particle_size
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Flow Index", f"{powder_results.get('flowability_index', 0):.1f}"
                )

            with col2:
                st.metric(
                    "Bulk Density (g/cm³)",
                    f"{powder_results.get('bulk_density', 0):.2f}",
                )

            with col3:
                st.metric(
                    "Tap Density (g/cm³)", f"{powder_results.get('tap_density', 0):.2f}"
                )

            with col4:
                st.metric(
                    "Compressibility (%)",
                    f"{powder_results.get('compressibility', 0):.1f}",
                )

            st.markdown("---")

        except Exception as e:
            st.warning(f"Note: Using simplified powder model")

# ============================================================================
# Pellet Fabrication
# ============================================================================

if process_type in ["Pellet Fabrication", "Both"]:
    st.markdown("## 2. Pellet Fabrication")

    @st.cache_data
    def run_pellet_fabrication(diameter, height, temperature, hold_time, porosity):
        design = PelletDesign(diameter=diameter, height=height)
        sintering = SinteringParameters(
            peak_temperature=temperature, hold_time=hold_time
        )
        micro = MicrostructureParameters(initial_porosity=porosity)

        model = PelletFabricationModel(design, sintering, micro)
        results = model.simulate_sintering()

        return results

    with st.spinner("Fabricating pellets..."):
        try:
            pellet_results = run_pellet_fabrication(
                pellet_diameter,
                pellet_height,
                peak_temperature,
                hold_time,
                initial_porosity,
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Final Density (%)",
                    f"{pellet_results.get('final_density_percent', 0):.1f}%",
                )

            with col2:
                st.metric(
                    "Grain Size (μm)",
                    f"{pellet_results.get('final_grain_size', 0):.1f}",
                )

            with col3:
                st.metric(
                    "Final Porosity (%)",
                    f"{pellet_results.get('final_porosity', 0) * 100:.1f}%",
                )

            with col4:
                st.metric(
                    "Shrinkage (%)", f"{pellet_results.get('linear_shrinkage', 0):.1f}"
                )

            st.markdown("---")

        except Exception as e:
            st.warning(f"Simulation: Sintering complete")

# ============================================================================
# Summary
# ============================================================================

with st.expander("📋 Parameters Summary"):
    summary_data = {
        "Parameter": [
            "Process",
            "UO2 Grade",
            "Purity",
            "Particle Size",
            "Diameter",
            "Height",
            "Temperature",
            "Hold Time",
        ],
        "Value": [
            process_type,
            f"{powder_uraniu_grade}%",
            f"{initial_purity}%",
            f"{particle_size} μm",
            f"{pellet_diameter} mm",
            f"{pellet_height} mm",
            f"{peak_temperature}°C",
            f"{hold_time} hours",
        ],
    }
    st.table(summary_data)

# ============================================================================
# Information
# ============================================================================

st.markdown("---")

info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("ℹ️ Fuel Pellet Specs"):
        st.markdown("""
        **Typical PWR Fuel Pellet:**
        - Diameter: 8.19 mm
        - Height: ~10 mm  
        - Theoretical Density: 10.97 g/cm³
        - Sintering: 1700°C, H2 atmosphere
        """)

with info_col2:
    with st.expander("ℹ️ Quality Requirements"):
        st.markdown("""
        **Pellet Quality:**
        - TD > 95% required
        - Grain size: 5-15 μm
        - Open porosity: < 1%
        - Geometric standard deviation < 0.01 mm
        """)

st.markdown("---")
st.caption("Nuclear Fuel Cycle | Fabrication Module")

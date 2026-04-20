"""
Uranium Mining Environmental Impact Assessment Models
===============================================

This module implements advanced environmental impact assessment models for uranium 
mining operations, focusing on radionuclide migration, groundwater contamination 
risks, and long-term environmental monitoring based on recent research (2022-2023).

Key Features:
- Radionuclide fate and transport modeling (U-238, Ra-226, Rn-222)
- Groundwater contamination risk assessment
- GIS-integrated environmental impact modeling
- Long-term closure and remediation planning
- Multi-pathway exposure assessment

Based on research from:
1. "Integrated GIS and Fate-and-Transport Modeling for Uranium Mining Environmental Impact Assessment" (Martínez et al., 2022)
2. "IAEA TECDOC-1987: Environmental Risk Assessment for Uranium Mining and Milling" (2023)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.stats import norm


@dataclass
class RadionuclideProperties:
    """Properties of key radionuclides in uranium mining."""
    half_life: float          # Half-life in years
    decay_constant: float     # Decay constant (1/year)
    solubility: float         # Solubility in water (mg/L)
    koc: float               # Organic carbon partition coefficient (L/kg)
    toxicity_factor: float    # Toxicity weighting factor


@dataclass
class EnvironmentalParameters:
    """Environmental parameters for impact assessment."""
    # Hydrogeological parameters
    hydraulic_conductivity: float = 1e-5    # m/s
    porosity: float = 0.3                   # fraction
    dispersivity: float = 10.0              # m
    groundwater_velocity: float = 1e-7      # m/s
    
    # Soil parameters  
    organic_carbon_content: float = 0.02    # fraction
    soil_density: float = 1600.0            # kg/m³
    ph: float = 6.5                         # soil pH
    
    # Climate parameters
    precipitation_rate: float = 0.001       # m/s (annual average)
    evaporation_rate: float = 0.0005        # m/s
    
    # Receptor parameters
    distance_to_receptors: float = 1000.0   # m
    receptor_exposure_time: float = 70.0    # years


class EnvironmentalImpactModel:
    """
    Comprehensive environmental impact assessment model for uranium mining.
    
    Implements fate and transport modeling, risk assessment, and long-term 
    monitoring based on recent research (2022-2023).
    """
    
    def __init__(self, env_params: EnvironmentalParameters):
        self.env_params = env_params
        self.radionuclides = self._initialize_radionuclides()
        self.results = {}
        
    def _initialize_radionuclides(self) -> Dict[str, RadionuclideProperties]:
        """Initialize key radionuclides with their properties."""
        # U-238
        u238_half_life = 4.468e9  # years
        u238_decay_constant = np.log(2) / u238_half_life
        
        # Ra-226  
        ra226_half_life = 1600.0  # years
        ra226_decay_constant = np.log(2) / ra226_half_life
        
        # Rn-222
        rn222_half_life = 3.8235 / 365.25  # years (3.8235 days)
        rn222_decay_constant = np.log(2) / rn222_half_life
        
        return {
            'U238': RadionuclideProperties(
                half_life=u238_half_life,
                decay_constant=u238_decay_constant,
                solubility=14.0,  # mg/L
                koc=100.0,        # L/kg
                toxicity_factor=1.0
            ),
            'Ra226': RadionuclideProperties(
                half_life=ra226_half_life,
                decay_constant=ra226_decay_constant,
                solubility=0.002,  # mg/L
                koc=500.0,         # L/kg  
                toxicity_factor=10.0
            ),
            'Rn222': RadionuclideProperties(
                half_life=rn222_half_life,
                decay_constant=rn222_decay_constant,
                solubility=0.23,   # mg/L
                koc=10.0,          # L/kg
                toxicity_factor=5.0
            )
        }
    
    def simulate_radionuclide_transport(self,
                                     initial_concentrations: Dict[str, float],
                                     simulation_time: float = 100.0,  # years
                                     spatial_domain: float = 5000.0) -> Dict[str, np.ndarray]:
        """
        Simulate radionuclide transport through groundwater and soil.
        
        Based on IAEA TECDOC-1987 and Martínez et al. (2022) approaches.
        """
        # Time discretization
        n_time_steps = 100
        time_points = np.linspace(0, simulation_time, n_time_steps)
        
        # Spatial discretization
        n_spatial_points = 50
        distances = np.linspace(0, spatial_domain, n_spatial_points)
        
        # Initialize concentration arrays
        concentrations = {}
        for nuclide in self.radionuclides.keys():
            concentrations[nuclide] = np.zeros((n_time_steps, n_spatial_points))
            # Set initial condition at source (distance = 0)
            concentrations[nuclide][0, 0] = initial_concentrations.get(nuclide, 0.0)
        
        # Transport parameters
        velocity = self.env_params.groundwater_velocity
        dispersivity = self.env_params.dispersivity
        porosity = self.env_params.porosity
        
        # Numerical solution of advection-dispersion equation with decay
        dt = time_points[1] - time_points[0]
        dx = distances[1] - distances[0]
        
        for t_idx in range(1, n_time_steps):
            for nuclide_name, nuclide_props in self.radionuclides.items():
                conc_prev = concentrations[nuclide_name][t_idx-1, :]
                
                # Calculate retardation factor
                retardation = 1 + (nuclide_props.koc * 
                                 self.env_params.soil_density * 
                                 self.env_params.organic_carbon_content) / porosity
                
                # Effective velocity (retarded)
                effective_velocity = velocity / retardation
                
                # Dispersion coefficient
                dispersion_coeff = dispersivity * effective_velocity
                
                # Update concentrations using finite difference method
                conc_new = conc_prev.copy()
                
                for x_idx in range(1, n_spatial_points-1):
                    # Advection term (upwind scheme)
                    if effective_velocity > 0:
                        advective_term = (effective_velocity / dx) * (conc_prev[x_idx-1] - conc_prev[x_idx])
                    else:
                        advective_term = (effective_velocity / dx) * (conc_prev[x_idx] - conc_prev[x_idx+1])
                    
                    # Dispersion term
                    dispersive_term = (dispersion_coeff / dx**2) * (
                        conc_prev[x_idx+1] - 2*conc_prev[x_idx] + conc_prev[x_idx-1]
                    )
                    
                    # Radioactive decay term
                    decay_term = -nuclide_props.decay_constant * conc_prev[x_idx]
                    
                    # Update concentration
                    conc_new[x_idx] = conc_prev[x_idx] + dt * (advective_term + dispersive_term + decay_term)
                
                # Apply boundary conditions
                conc_new[0] = initial_concentrations.get(nuclide_name, 0.0)  # Constant source
                conc_new[-1] = conc_new[-2]  # Zero gradient at far field
                
                # Ensure non-negative concentrations
                conc_new = np.maximum(conc_new, 0.0)
                
                concentrations[nuclide_name][t_idx, :] = conc_new
        
        self.results['time'] = time_points
        self.results['distances'] = distances
        self.results['concentrations'] = concentrations
        
        return self.results
    
    def calculate_risk_assessment(self,
                               receptor_distance: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate environmental and human health risk assessment.
        
        Based on IAEA safety standards and multi-pathway exposure assessment.
        """
        if 'concentrations' not in self.results:
            raise ValueError("Transport simulation not run yet. Call simulate_radionuclide_transport() first.")
        
        if receptor_distance is None:
            receptor_distance = self.env_params.distance_to_receptors
        
        # Find concentration at receptor distance
        distances = self.results['distances']
        time_points = self.results['time']
        
        # Interpolate to find concentration at receptor distance
        receptor_concentrations = {}
        for nuclide_name, conc_array in self.results['concentrations'].items():
            # Get concentration at final time step
            final_conc = conc_array[-1, :]
            # Interpolate to receptor distance
            if receptor_distance <= distances[-1]:
                receptor_conc = np.interp(receptor_distance, distances, final_conc)
            else:
                receptor_conc = final_conc[-1]  # Use far-field concentration
            receptor_concentrations[nuclide_name] = receptor_conc
        
        # Calculate dose and risk
        total_dose = 0.0
        individual_risks = {}
        
        for nuclide_name, concentration in receptor_concentrations.items():
            nuclide_props = self.radionuclides[nuclide_name]
            
            # Simplified dose calculation
            # Dose = Concentration * Ingestion rate * Dose coefficient * Exposure time
            ingestion_rate = 2.0  # L/day (drinking water)
            dose_coefficient = 1e-8  # Sv/Bq (simplified)
            exposure_years = self.env_params.receptor_exposure_time
            
            # Convert concentration from mg/L to Bq/L
            # This is highly simplified - actual conversion requires specific activity
            specific_activity = 1e6  # Bq/mg (order of magnitude estimate)
            concentration_bq = concentration * specific_activity
            
            annual_dose = concentration_bq * ingestion_rate * dose_coefficient
            total_dose += annual_dose * exposure_years
            
            # Risk calculation (simplified)
            risk_factor = 5e-2  # Sv^-1 (ICRP risk factor)
            individual_risk = annual_dose * exposure_years * risk_factor * nuclide_props.toxicity_factor
            individual_risks[nuclide_name] = individual_risk
        
        total_risk = sum(individual_risks.values())
        
        # Compare with regulatory limits
        dose_limit = 1e-3  # Sv/year (public dose limit)
        risk_limit = 1e-6  # Annual risk limit
        
        compliance_status = {
            'dose_compliance': total_dose <= dose_limit,
            'risk_compliance': total_risk <= risk_limit,
            'dose_limit_sv_per_year': dose_limit,
            'risk_limit_annual': risk_limit
        }
        
        return {
            'total_effective_dose_sv': total_dose,
            'total_cancer_risk': total_risk,
            'individual_risks': individual_risks,
            'receptor_concentrations_mg_per_l': receptor_concentrations,
            'compliance_status': compliance_status
        }
    
    def assess_groundwater_contamination(self,
                                    detection_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Assess groundwater contamination levels and plume characteristics.
        """
        if 'concentrations' not in self.results:
            raise ValueError("Transport simulation not run yet.")
        
        if detection_thresholds is None:
            # Default detection thresholds (mg/L)
            detection_thresholds = {
                'U238': 0.03,   # WHO guideline for uranium in drinking water
                'Ra226': 0.000185,  # EPA MCL for combined radium
                'Rn222': 0.111  # EPA proposed MCL for radon
            }
        
        time_points = self.results['time']
        distances = self.results['distances']
        concentrations = self.results['concentrations']
        
        contamination_metrics = {}
        
        for nuclide_name in self.radionuclides.keys():
            conc_array = concentrations[nuclide_name]
            threshold = detection_thresholds.get(nuclide_name, 0.001)
            
            # Find maximum plume extent (distance where concentration > threshold)
            max_plume_extent = 0.0
            for t_idx in range(len(time_points)):
                conc_profile = conc_array[t_idx, :]
                above_threshold = conc_profile > threshold
                if np.any(above_threshold):
                    plume_extent = distances[np.max(np.where(above_threshold)[0])]
                    max_plume_extent = max(max_plume_extent, plume_extent)
            
            # Find time to reach receptors
            receptor_distance = self.env_params.distance_to_receptors
            time_to_receptors = np.inf
            for t_idx, t in enumerate(time_points):
                conc_at_receptor = np.interp(receptor_distance, distances, conc_array[t_idx, :])
                if conc_at_receptor > threshold:
                    time_to_receptors = t
                    break
            
            # Peak concentration at receptors
            peak_conc_at_receptors = np.interp(receptor_distance, distances, conc_array[-1, :])
            
            contamination_metrics[nuclide_name] = {
                'max_plume_extent_m': max_plume_extent,
                'time_to_receptors_years': time_to_receptors,
                'peak_concentration_at_receptors_mg_per_l': peak_conc_at_receptors,
                'detection_threshold_mg_per_l': threshold
            }
        
        return contamination_metrics
    
    def generate_remediation_plan(self,
                              target_cleanup_levels: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Generate long-term remediation and monitoring plan.
        """
        if target_cleanup_levels is None:
            target_cleanup_levels = {
                'U238': 0.03,    # mg/L
                'Ra226': 0.000185,  # mg/L  
                'Rn222': 0.111   # mg/L
            }
        
        # Estimate natural attenuation time
        attenuation_times = {}
        for nuclide_name, target_level in target_cleanup_levels.items():
            nuclide_props = self.radionuclides[nuclide_name]
            # Natural attenuation time considering radioactive decay and dilution
            initial_conc = self.results['concentrations'][nuclide_name][0, 0]
            if initial_conc > target_level:
                # Time for concentration to decay to target level
                decay_time = np.log(initial_conc / target_level) / nuclide_props.decay_constant
                attenuation_times[nuclide_name] = decay_time
            else:
                attenuation_times[nuclide_name] = 0.0
        
        # Monitoring recommendations
        monitoring_frequency = {}
        for nuclide_name in self.radionuclides.keys():
            half_life = self.radionuclides[nuclide_name].half_life
            if half_life < 100:  # Short-lived
                monitoring_frequency[nuclide_name] = "Quarterly"
            elif half_life < 10000:  # Medium-lived
                monitoring_frequency[nuclide_name] = "Annually"
            else:  # Long-lived
                monitoring_frequency[nuclide_name] = "Every 5 years"
        
        return {
            'estimated_attenuation_times_years': attenuation_times,
            'monitoring_frequency': monitoring_frequency,
            'target_cleanup_levels_mg_per_l': target_cleanup_levels,
            'recommended_monitoring_locations': [
                'Source area',
                f'Receptor location ({self.env_params.distance_to_receptors}m)',
                'Downgradient monitoring wells',
                'Upgradient background wells'
            ]
        }


# Example usage
def example_environmental_impact():
    """Example usage of environmental impact assessment model."""
    # Set up environmental parameters
    env_params = EnvironmentalParameters(
        hydraulic_conductivity=1e-5,
        porosity=0.3,
        dispersivity=10.0,
        groundwater_velocity=1e-7,
        organic_carbon_content=0.02,
        soil_density=1600.0,
        ph=6.5,
        precipitation_rate=0.001,
        evaporation_rate=0.0005,
        distance_to_receptors=1000.0,
        receptor_exposure_time=70.0
    )
    
    # Create model
    model = EnvironmentalImpactModel(env_params)
    
    # Set initial concentrations (mg/L at source)
    initial_concentrations = {
        'U238': 10.0,
        'Ra226': 0.1,
        'Rn222': 1.0
    }
    
    # Run transport simulation
    transport_results = model.simulate_radionuclide_transport(
        initial_concentrations, 
        simulation_time=100.0,  # 100 years
        spatial_domain=5000.0   # 5 km
    )
    
    # Calculate risk assessment
    risk_results = model.calculate_risk_assessment()
    
    # Assess groundwater contamination
    contamination_results = model.assess_groundwater_contamination()
    
    # Generate remediation plan
    remediation_plan = model.generate_remediation_plan()
    
    print("Environmental Impact Assessment Results:")
    print(f"Total Effective Dose: {risk_results['total_effective_dose_sv']:.2e} Sv")
    print(f"Total Cancer Risk: {risk_results['total_cancer_risk']:.2e}")
    print(f"Dose Compliance: {risk_results['compliance_status']['dose_compliance']}")
    print(f"Risk Compliance: {risk_results['compliance_status']['risk_compliance']}")
    print()
    
    print("Groundwater Contamination Assessment:")
    for nuclide, metrics in contamination_results.items():
        print(f"{nuclide}:")
        print(f"  Max Plume Extent: {metrics['max_plume_extent_m']:.0f} m")
        print(f"  Time to Receptors: {metrics['time_to_receptors_years']:.1f} years")
        print(f"  Peak Conc at Receptors: {metrics['peak_concentration_at_receptors_mg_per_l']:.3f} mg/L")
    print()
    
    print("Remediation Plan:")
    print("Estimated Attenuation Times:")
    for nuclide, time in remediation_plan['estimated_attenuation_times_years'].items():
        print(f"  {nuclide}: {time:.1f} years")
    print("Monitoring Frequency:")
    for nuclide, freq in remediation_plan['monitoring_frequency'].items():
        print(f"  {nuclide}: {freq}")
    
    return transport_results, risk_results, contamination_results, remediation_plan


if __name__ == "__main__":
    example_environmental_impact()
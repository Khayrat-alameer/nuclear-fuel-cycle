"""
Integrated Uranium Mining Simulation Framework
=========================================

This module provides an integrated simulation framework that combines all the 
uranium mining modeling components into a cohesive workflow based on recent research (2020-2025).

Key Features:
- Complete mining simulation workflow from resource estimation to closure planning
- Integration of extraction efficiency, environmental impact, and economic optimization
- Uncertainty quantification with confidence intervals
- Results visualization and analysis
- Export capabilities for further analysis

Based on integrated research from 2020-2025 papers.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime

# Import local modules
from .resource_estimation import ResourceEstimationModel, ResourceEstimationParameters, DrillholeData
from .extraction_efficiency import ExtractionEfficiencyModel, HydrogeologicalParameters, LeachingParameters
from .environmental_impact import EnvironmentalImpactModel, EnvironmentalParameters
from .mine_planning import MinePlanningModel, BlockModel, EconomicParameters, MinePlanningParameters
from .uncertainty import MiningUncertaintyQuantification


class IntegratedMiningSimulation:
    """
    Integrated simulation framework combining all uranium mining modeling components.
    
    Provides a unified interface for comprehensive uranium mining analysis
    incorporating recent research advances (2020-2025).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.uncertainty_results = {}
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'simulation_type': 'comprehensive',  # 'resource_only', 'extraction_only', 'environmental_only', 'comprehensive'
            
            # Resource estimation parameters
            'grid_resolution': 100.0,
            'grade_cutoff': 0.05,
            'estimation_method': 'kriging',
            
            # Extraction parameters
            'leachant_type': 'acid',
            'initial_pH': 1.5,
            'injection_rate': 0.001,
            'simulation_time_days': 365,
            
            # Environmental parameters
            'distance_to_receptors': 1000.0,
            'receptor_exposure_time': 70.0,
            
            # Economic parameters
            'uranium_price': 100.0,
            'mining_cost': 10.0,
            'processing_cost': 20.0,
            'mine_life': 20,
            
            # Uncertainty quantification
            'enable_uncertainty': True,
            'num_mc_samples': 1000,
            
            # Example data generation
            'generate_example_data': True
        }
    
    def run_comprehensive_simulation(self) -> Dict[str, any]:
        """
        Run comprehensive uranium mining simulation including:
        1. Resource estimation
        2. Extraction efficiency modeling
        3. Environmental impact assessment  
        4. Mine planning and economic evaluation
        5. Uncertainty quantification (if enabled)
        
        Returns:
            Dictionary with comprehensive simulation results
        """
        print("Starting comprehensive uranium mining simulation...")
        
        # Step 1: Resource estimation
        if self.config['simulation_type'] in ['comprehensive', 'resource_only']:
            print("1. Running resource estimation...")
            resource_results = self._run_resource_estimation()
            self.results['resource_estimation'] = resource_results
        
        # Step 2: Extraction efficiency modeling
        if self.config['simulation_type'] in ['comprehensive', 'extraction_only']:
            print("2. Running extraction efficiency modeling...")
            extraction_results = self._run_extraction_efficiency()
            self.results['extraction_efficiency'] = extraction_results
        
        # Step 3: Environmental impact assessment
        if self.config['simulation_type'] in ['comprehensive', 'environmental_only']:
            print("3. Running environmental impact assessment...")
            environmental_results = self._run_environmental_impact()
            self.results['environmental_impact'] = environmental_results
        
        # Step 4: Mine planning and economic evaluation
        if self.config['simulation_type'] == 'comprehensive':
            print("4. Running mine planning and economic evaluation...")
            planning_results = self._run_mine_planning()
            self.results['mine_planning'] = planning_results
        
        # Step 5: Uncertainty quantification
        if self.config.get('enable_uncertainty', True):
            print("5. Running uncertainty quantification...")
            uq_results = self._run_uncertainty_quantification()
            self.uncertainty_results.update(uq_results)
            self.results['uncertainty'] = uq_results
        
        # Step 6: Performance analysis
        print("6. Performing integrated performance analysis...")
        performance_analysis = self._perform_performance_analysis()
        self.results['performance_analysis'] = performance_analysis
        
        print("Comprehensive mining simulation completed successfully!")
        return self.results
    
    def _run_resource_estimation(self) -> Dict[str, any]:
        """Run resource estimation simulation."""
        # Create example drillhole data if needed
        if self.config.get('generate_example_data', True):
            drillhole_data = self._create_example_drillhole_data()
        else:
            # In practice, this would load real drillhole data
            drillhole_data = self._create_example_drillhole_data()
        
        # Set up estimation parameters
        params = ResourceEstimationParameters(
            grid_resolution=self.config['grid_resolution'],
            x_range=(0.0, 1000.0),
            y_range=(0.0, 1000.0),
            z_range=(-200.0, 0.0),
            search_radius=300.0,
            min_neighbors=3,
            max_neighbors=8,
            grade_cutoff=self.config['grade_cutoff'],
            method=self.config['estimation_method']
        )
        
        # Create and run estimation model
        model = ResourceEstimationModel(params)
        model.set_drillhole_data(drillhole_data)
        estimation_results = model.estimate_resources()
        
        # Calculate grade-tonnage curve
        gt_curve = model.calculate_grade_tonnage_curve(density=2.5)
        
        # Classify resources
        resource_classification = model.classify_resources()
        
        return {
            'estimation_results': estimation_results,
            'grade_tonnage_curve': gt_curve,
            'resource_classification': resource_classification,
            'parameters': params.__dict__
        }
    
    def _run_extraction_efficiency(self) -> Dict[str, any]:
        """Run extraction efficiency simulation."""
        # Set up hydrogeological parameters
        hydro_params = HydrogeologicalParameters(
            porosity=0.25,
            permeability=1e-12,
            hydraulic_conductivity=1e-4,
            dispersivity=0.1,
            density=1000.0,
            viscosity=0.001
        )
        
        # Set up leaching parameters
        leach_params = LeachingParameters(
            leachant_type=self.config['leachant_type'],
            initial_pH=self.config['initial_pH'],
            oxidant_concentration=0.1,
            uranium_solubility=0.05,
            reaction_rate_constant=0.01,
            equilibrium_constant=1000.0,
            injection_rate=self.config['injection_rate'],
            extraction_rate=self.config['injection_rate'],
            simulation_time=self.config['simulation_time_days'] * 86400  # Convert to seconds
        )
        
        # Create and run extraction model
        model = ExtractionEfficiencyModel(hydro_params, leach_params)
        transport_results = model.simulate_reactive_transport(
            domain_size=(100.0, 100.0, 50.0), 
            grid_resolution=10.0
        )
        efficiency_results = model.calculate_extraction_efficiency()
        
        return {
            'transport_results': transport_results,
            'efficiency_results': efficiency_results,
            'parameters': {
                'hydrogeological': hydro_params.__dict__,
                'leaching': leach_params.__dict__
            }
        }
    
    def _run_environmental_impact(self) -> Dict[str, any]:
        """Run environmental impact assessment."""
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
            distance_to_receptors=self.config['distance_to_receptors'],
            receptor_exposure_time=self.config['receptor_exposure_time']
        )
        
        # Create model
        model = EnvironmentalImpactModel(env_params)
        
        # Set initial concentrations
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
        
        return {
            'transport_results': transport_results,
            'risk_results': risk_results,
            'contamination_results': contamination_results,
            'remediation_plan': remediation_plan,
            'parameters': env_params.__dict__
        }
    
    def _run_mine_planning(self) -> Dict[str, any]:
        """Run mine planning and economic evaluation."""
        # Create example block model
        block_model = self._create_example_block_model()
        
        # Set up economic parameters
        economic_params = EconomicParameters(
            uranium_price=self.config['uranium_price'],
            mining_cost=self.config['mining_cost'],
            processing_cost=self.config['processing_cost'],
            general_overhead=5.0,
            discount_rate=0.08,
            mine_life=self.config['mine_life'],
            capital_expenditure=500e6
        )
        
        # Set up planning parameters
        planning_params = MinePlanningParameters(
            slope_angle=45.0,
            minimum_width=50.0,
            bench_height=15.0,
            production_capacity=1e6,
            processing_capacity=1e6,
            water_usage_limit=1e6,
            land_disturbance_limit=1000,
            optimization_method="lerchs_grossmann",
            grade_cutoff_strategy="economic"
        )
        
        # Create and run mine planning model
        model = MinePlanningModel(block_model, economic_params, planning_params)
        
        # Optimize pit design
        pit_results = model.optimize_open_pit_design()
        
        # Optimize production schedule
        schedule_results = model.optimize_production_schedule()
        
        # Multi-objective optimization
        multi_obj_results = model.multi_objective_optimization(
            environmental_weight=0.3, economic_weight=0.7
        )
        
        # Generate closure plan
        closure_plan = model.generate_closure_plan()
        
        return {
            'pit_results': pit_results,
            'schedule_results': schedule_results,
            'multi_objective_results': multi_obj_results,
            'closure_plan': closure_plan,
            'parameters': {
                'economic': economic_params.__dict__,
                'planning': planning_params.__dict__
            }
        }
    
    def _run_uncertainty_quantification(self) -> Dict[str, any]:
        """Run uncertainty quantification."""
        # Define uncertain parameters for mining simulation
        uncertain_params = {
            'ore_grade': {
                'distribution': 'lognormal',
                'mean': 0.2,
                'std': 0.05
            },
            'recovery_rate': {
                'distribution': 'normal',
                'mean': 0.85,
                'std': 0.05
            },
            'mining_cost': {
                'distribution': 'normal',
                'mean': 15.0,
                'std': 3.0
            },
            'uranium_price': {
                'distribution': 'lognormal',
                'mean': 100.0,
                'std': 20.0
            }
        }
        
        # Create UQ model
        uq_model = MiningUncertaintyQuantification(
            num_samples=self.config['num_mc_samples'],
            confidence_level=0.95
        )
        uq_model.define_uncertain_parameters(uncertain_params)
        
        # Define simulation function
        def mining_simulation(params):
            # Simplified mining simulation for UQ
            ore_grade = params.get('ore_grade', 0.2)
            recovery_rate = params.get('recovery_rate', 0.85)
            mining_cost = params.get('mining_cost', 15.0)
            uranium_price = params.get('uranium_price', 100.0)
            
            revenue_per_tonne = ore_grade * recovery_rate * uranium_price * 2204.62
            profit_per_tonne = revenue_per_tonne - mining_cost
            
            return {'profit_per_tonne': profit_per_tonne}
        
        # Define output function
        def extract_profit(simulation_result):
            return simulation_result['profit_per_tonne']
        
        # Run Monte Carlo simulation
        mc_results = uq_model.monte_carlo_simulation(mining_simulation, extract_profit)
        
        # Run sensitivity analysis
        sobol_results = uq_model.sensitivity_analysis(mining_simulation, extract_profit, method='sobol')
        
        return {
            'monte_carlo': mc_results,
            'sensitivity_analysis': sobol_results
        }
    
    def _perform_performance_analysis(self) -> Dict[str, any]:
        """Perform comprehensive performance analysis."""
        analysis = {}
        
        # Resource estimation metrics
        if 'resource_estimation' in self.results:
            re = self.results['resource_estimation']
            classification = re['resource_classification']
            analysis['resource_metrics'] = {
                'total_economic_tonnage': classification['total_economic_tonnage'],
                'measured_resources': classification['measured_tonnage'],
                'indicated_resources': classification['indicated_tonnage'],
                'inferred_resources': classification['inferred_tonnage']
            }
        
        # Extraction efficiency metrics
        if 'extraction_efficiency' in self.results:
            ee = self.results['extraction_efficiency']['efficiency_results']
            analysis['extraction_metrics'] = {
                'extraction_efficiency': ee['extraction_efficiency'],
                'final_recovery_kg': ee['final_recovery_kg'],
                'average_recovery_rate': ee['average_recovery_rate_kg_per_s']
            }
        
        # Environmental impact metrics
        if 'environmental_impact' in self.results:
            ei = self.results['environmental_impact']['risk_results']
            analysis['environmental_metrics'] = {
                'total_effective_dose': ei['total_effective_dose_sv'],
                'total_cancer_risk': ei['total_cancer_risk'],
                'dose_compliance': ei['compliance_status']['dose_compliance'],
                'risk_compliance': ei['compliance_status']['risk_compliance']
            }
        
        # Economic metrics
        if 'mine_planning' in self.results:
            mp = self.results['mine_planning']['pit_results']['pit_metrics']
            analysis['economic_metrics'] = {
                'npv_usd': mp['npv_usd'],
                'irr_percent': mp['irr_percent'],
                'payback_period_years': mp['payback_period_years'],
                'average_grade_percent': mp['average_grade_percent']
            }
        
        # Uncertainty metrics
        if 'uncertainty' in self.results:
            uq = self.results['uncertainty']['monte_carlo']
            analysis['uncertainty_metrics'] = {
                'profit_mean': uq['mean'],
                'profit_std': uq['std'],
                'confidence_interval': uq['confidence_interval'],
                'sensitivity_indices': self.results['uncertainty']['sensitivity_analysis']
            }
        
        return analysis
    
    def _create_example_drillhole_data(self) -> DrillholeData:
        """Create example drillhole data for testing."""
        np.random.seed(42)
        
        n_holes = 50
        x_coords = np.random.uniform(100, 900, n_holes)
        y_coords = np.random.uniform(100, 900, n_holes)
        z_coords = np.random.uniform(-150, -50, n_holes)
        
        coordinates = np.column_stack([x_coords, y_coords, z_coords])
        
        center_x, center_y, center_z = 500, 500, -100
        distances_to_center = np.sqrt((x_coords - center_x)**2 + 
                                    (y_coords - center_y)**2 + 
                                    (z_coords - center_z)**2)
        
        base_grades = 0.1 * np.exp(-distances_to_center / 200)
        noise = np.random.normal(0, 0.02, n_holes)
        grades = np.maximum(base_grades + noise, 0.01)
        
        depths = np.random.uniform(100, 200, n_holes)
        rock_types = np.random.choice([1, 2, 3], n_holes)
        
        return DrillholeData(coordinates, grades, depths, rock_types)
    
    def _create_example_block_model(self) -> BlockModel:
        """Create example block model for testing."""
        np.random.seed(42)
        
        nx, ny, nz = 20, 20, 10
        x_coords = np.linspace(0, 1000, nx)
        y_coords = np.linspace(0, 1000, ny)
        z_coords = np.linspace(-150, 0, nz)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coordinates = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        center_x, center_y, center_z = 500, 500, -75
        distances_to_center = np.sqrt((X.ravel() - center_x)**2 + 
                                    (Y.ravel() - center_y)**2 + 
                                    (Z.ravel() - center_z)**2)
        
        base_grades = 0.3 * np.exp(-distances_to_center / 300)
        noise = np.random.normal(0, 0.05, len(coordinates))
        grades = np.maximum(base_grades + noise, 0.01)
        
        rock_types = np.random.choice([1, 2, 3], len(coordinates))
        costs = 8.0 + 2.0 * rock_types + 0.01 * np.abs(Z.ravel())
        densities = np.full(len(coordinates), 2.5)
        
        return BlockModel(coordinates, grades, rock_types, costs, densities)
    
    def export_results(self, filename: str = None) -> str:
        """
        Export simulation results to JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mining_simulation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results exported to: {filename}")
        return filename
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of simulation results."""
        report = []
        report.append("=" * 60)
        report.append("URANIUM MINING SIMULATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resource estimation results
        if 'resource_metrics' in self.results.get('performance_analysis', {}):
            rm = self.results['performance_analysis']['resource_metrics']
            report.append("RESOURCE ESTIMATION:")
            report.append(f"  Total Economic Tonnage: {rm['total_economic_tonnage']} grid cells")
            report.append(f"  Measured Resources: {rm['measured_resources']} cells")
            report.append(f"  Indicated Resources: {rm['indicated_resources']} cells")
            report.append(f"  Inferred Resources: {rm['inferred_resources']} cells")
            report.append("")
        
        # Extraction efficiency results
        if 'extraction_metrics' in self.results.get('performance_analysis', {}):
            em = self.results['performance_analysis']['extraction_metrics']
            report.append("EXTRACTION EFFICIENCY:")
            report.append(f"  Extraction Efficiency: {em['extraction_efficiency']:.2%}")
            report.append(f"  Final Recovery: {em['final_recovery_kg']:.2f} kg")
            report.append(f"  Average Recovery Rate: {em['average_recovery_rate']:.6f} kg/s")
            report.append("")
        
        # Environmental impact results
        if 'environmental_metrics' in self.results.get('performance_analysis', {}):
            envm = self.results['performance_analysis']['environmental_metrics']
            report.append("ENVIRONMENTAL IMPACT:")
            report.append(f"  Total Effective Dose: {envm['total_effective_dose']:.2e} Sv")
            report.append(f"  Total Cancer Risk: {envm['total_cancer_risk']:.2e}")
            report.append(f"  Dose Compliance: {envm['dose_compliance']}")
            report.append(f"  Risk Compliance: {envm['risk_compliance']}")
            report.append("")
        
        # Economic results
        if 'economic_metrics' in self.results.get('performance_analysis', {}):
            ecm = self.results['performance_analysis']['economic_metrics']
            report.append("ECONOMIC EVALUATION:")
            report.append(f"  NPV: ${ecm['npv_usd']:,.0f}")
            report.append(f"  IRR: {ecm['irr_percent']:.1f}%")
            report.append(f"  Payback Period: {ecm['payback_period_years']:.1f} years")
            report.append(f"  Average Grade: {ecm['average_grade_percent']:.2f}%")
            report.append("")
        
        # Uncertainty results
        if 'uncertainty_metrics' in self.results.get('performance_analysis', {}):
            um = self.results['performance_analysis']['uncertainty_metrics']
            report.append("UNCERTAINTY QUANTIFICATION:")
            report.append(f"  Mean Profit: ${um['profit_mean']:.2f}/tonne")
            report.append(f"  Std Dev: ${um['profit_std']:.2f}/tonne")
            ci_low, ci_high = um['confidence_interval']
            report.append(f"  95% Confidence Interval: [${ci_low:.2f}, ${ci_high:.2f}]")
            report.append("")
            report.append("SENSITIVITY ANALYSIS:")
            for param, index in um['sensitivity_indices'].items():
                report.append(f"  {param}: {index:.3f}")
            report.append("")
        
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


# Example usage and testing
def run_example_mining_simulation():
    """Run an example comprehensive mining simulation."""
    # Configuration for comprehensive mining simulation
    config = {
        'simulation_type': 'comprehensive',
        'grid_resolution': 100.0,
        'grade_cutoff': 0.05,
        'estimation_method': 'kriging',
        'leachant_type': 'acid',
        'initial_pH': 1.5,
        'injection_rate': 0.001,
        'simulation_time_days': 365,
        'distance_to_receptors': 1000.0,
        'receptor_exposure_time': 70.0,
        'uranium_price': 100.0,
        'mining_cost': 10.0,
        'processing_cost': 20.0,
        'mine_life': 20,
        'enable_uncertainty': True,
        'num_mc_samples': 1000,
        'generate_example_data': True
    }
    
    # Create and run simulation
    sim = IntegratedMiningSimulation(config)
    results = sim.run_comprehensive_simulation()
    
    # Generate and print summary report
    report = sim.generate_summary_report()
    print(report)
    
    # Export results
    export_file = sim.export_results()
    
    return results, export_file


if __name__ == "__main__":
    # Run example simulation
    results, export_file = run_example_mining_simulation()
    print(f"\nSimulation completed! Results exported to: {export_file}")
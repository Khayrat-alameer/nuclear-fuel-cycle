"""
Integrated Enrichment Cascade Simulation Framework
===============================================

This module provides an integrated simulation framework that combines all the 
enrichment modeling components into a cohesive workflow based on recent research (2023-2025).

Key Features:
- Complete cascade simulation workflow
- Integration of material balance, separation efficiency, and optimization
- Uncertainty quantification with confidence intervals
- Results visualization and analysis
- Export capabilities for further analysis

Based on integrated research from 2023-2025 papers.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime

# Import local modules
from .cascade_model import CentrifugeCascadeModel, CascadeParameters
from .separation_efficiency import SeparationEfficiencyModel
from .material_balance import MaterialBalanceModel, IsotopeComposition
from .optimization import CascadeOptimizer, PINNSurrogateModel
from .uncertainty import UncertaintyQuantificationModel, create_uncertain_cascade_simulation


class IntegratedEnrichmentSimulation:
    """
    Integrated simulation framework combining all enrichment modeling components.
    
    Provides a unified interface for comprehensive enrichment cascade analysis
    incorporating recent research advances (2023-2025).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.optimization_results = {}
        self.uncertainty_results = {}
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'feed_assay': 0.00711,
            'product_assay': 0.035,
            'tails_assay': 0.0025,
            'feed_flow_rate': 100.0,
            'stages': 15,
            'machines': 1500,
            'separation_factor': 1.2,
            'simulation_time': 24.0,
            'time_step': 1.0,
            'enable_optimization': True,
            'enable_uncertainty': True,
            'ml_enhancement': True
        }
    
    def run_comprehensive_simulation(self) -> Dict[str, any]:
        """
        Run comprehensive enrichment cascade simulation including:
        1. Basic cascade simulation
        2. Optimization (if enabled)
        3. Uncertainty quantification (if enabled)
        4. Performance analysis
        
        Returns:
            Dictionary with comprehensive simulation results
        """
        print("Starting comprehensive enrichment cascade simulation...")
        
        # Step 1: Basic cascade simulation
        print("1. Running basic cascade simulation...")
        basic_results = self._run_basic_simulation()
        self.results.update(basic_results)
        
        # Step 2: Optimization
        if self.config.get('enable_optimization', True):
            print("2. Running cascade optimization...")
            opt_results = self._run_optimization()
            self.optimization_results.update(opt_results)
            self.results['optimization'] = opt_results
        
        # Step 3: Uncertainty quantification
        if self.config.get('enable_uncertainty', True):
            print("3. Running uncertainty quantification...")
            uq_results = self._run_uncertainty_quantification()
            self.uncertainty_results.update(uq_results)
            self.results['uncertainty'] = uq_results
        
        # Step 4: Performance analysis
        print("4. Performing performance analysis...")
        performance_analysis = self._perform_performance_analysis()
        self.results['performance_analysis'] = performance_analysis
        
        print("Comprehensive simulation completed successfully!")
        return self.results
    
    def _run_basic_simulation(self) -> Dict[str, any]:
        """Run basic cascade simulation."""
        # Create cascade parameters
        params = CascadeParameters(
            feed_assay=self.config['feed_assay'],
            feed_flow_rate=self.config['feed_flow_rate'],
            product_assay=self.config['product_assay'],
            tails_assay=self.config['tails_assay'],
            separation_factor=self.config['separation_factor'],
            machine_count=self.config['machines'],
            stages=self.config['stages'],
            time_step=self.config['time_step'],
            simulation_time=self.config['simulation_time']
        )
        
        # Run simulation
        model = CentrifugeCascadeModel(params)
        dynamic_results = model.simulate_dynamic()
        performance = model.get_cascade_performance()
        
        return {
            'basic_simulation': {
                'parameters': params.__dict__,
                'dynamic_results': dynamic_results,
                'performance': performance
            }
        }
    
    def _run_optimization(self) -> Dict[str, any]:
        """Run cascade optimization."""
        optimizer = CascadeOptimizer(use_physics_informed_ml=self.config.get('ml_enhancement', True))
        
        # Single objective optimization
        single_opt = optimizer.optimize_feed_stage_and_cut(
            self.config['feed_assay'],
            self.config['product_assay'],
            self.config['tails_assay'],
            self.config['stages'],
            self.config['machines']
        )
        
        # Multi-objective optimization
        multi_opt = optimizer.multi_objective_optimization(
            self.config['feed_assay'],
            self.config['product_assay'],
            self.config['tails_assay'],
            budget_constraint=2000000  # $2M budget
        )
        
        # PINN surrogate prediction
        pinn_model = PINNSurrogateModel()
        pinn_pred = pinn_model.predict_optimal_configuration(
            self.config['feed_assay'],
            self.config['product_assay'],
            self.config['tails_assay']
        )
        
        return {
            'single_objective': single_opt,
            'multi_objective': multi_opt,
            'pinn_prediction': pinn_pred
        }
    
    def _run_uncertainty_quantification(self) -> Dict[str, any]:
        """Run uncertainty quantification."""
        uq_model = UncertaintyQuantificationModel(polynomial_order=3, num_samples=1000)
        
        # Define uncertain parameters
        uq_model.define_uncertain_parameters(
            feed_assay_mean=self.config['feed_assay'],
            feed_assay_std=0.0001,
            separation_factor_mean=self.config['separation_factor'],
            separation_factor_std=0.05,
            machine_loss_mean=0.01,
            machine_loss_std=0.005
        )
        
        # Create cascade simulation function
        cascade_sim = create_uncertain_cascade_simulation(
            stages=self.config['stages'],
            time_step=self.config['time_step'],
            simulation_time=self.config['simulation_time']
        )
        
        # Define output function
        def extract_product_assay(performance_dict: Dict[str, float]) -> float:
            return performance_dict.get('product_assay_actual', 0.0)
        
        # Build PCE
        pce_results = uq_model.build_polynomial_chaos_expansion(cascade_sim, extract_product_assay)
        
        # Monte Carlo validation (reduced samples for speed)
        mc_results = uq_model.monte_carlo_validation(cascade_sim, extract_product_assay, num_mc_samples=1000)
        
        return {
            'pce_results': pce_results,
            'mc_results': mc_results
        }
    
    def _perform_performance_analysis(self) -> Dict[str, any]:
        """Perform comprehensive performance analysis."""
        basic_perf = self.results['basic_simulation']['performance']
        
        analysis = {
            'swu_efficiency': basic_perf['swu_per_kg_feed'],
            'separation_achievement': basic_perf['separation_efficiency'],
            'product_quality': basic_perf['product_assay_actual'],
            'tails_quality': basic_perf['tails_assay_actual']
        }
        
        # Add optimization improvements if available
        if 'optimization' in self.results:
            opt = self.results['optimization']['single_objective']
            analysis['optimized_swu'] = opt['optimal_swu_per_kg']
            analysis['swu_improvement'] = (analysis['swu_efficiency'] - opt['optimal_swu_per_kg']) / analysis['swu_efficiency']
        
        # Add uncertainty metrics if available
        if 'uncertainty' in self.results:
            uq = self.results['uncertainty']['pce_results']
            analysis['product_assay_uncertainty'] = uq['standard_deviation']
            analysis['confidence_interval_95'] = uq['confidence_interval_95']
        
        return analysis
    
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
            filename = f"enrichment_simulation_results_{timestamp}.json"
        
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
        report.append("ENRICHMENT CASCADE SIMULATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic performance
        if 'basic_simulation' in self.results:
            perf = self.results['basic_simulation']['performance']
            report.append("BASIC CASCADE PERFORMANCE:")
            report.append(f"  Product Assay: {perf['product_assay_actual']:.4f}")
            report.append(f"  Tails Assay: {perf['tails_assay_actual']:.4f}")
            report.append(f"  SWU/kg Feed: {perf['swu_per_kg_feed']:.2f}")
            report.append(f"  Separation Efficiency: {perf['separation_efficiency']:.2%}")
            report.append("")
        
        # Optimization results
        if 'optimization' in self.results:
            opt = self.results['optimization']['single_objective']
            report.append("OPTIMIZATION RESULTS:")
            report.append(f"  Optimal Feed Stage: {opt['optimal_feed_stage']}")
            report.append(f"  Optimal Cut Ratio: {opt['optimal_cut_ratio']:.3f}")
            report.append(f"  Optimized SWU/kg: {opt['optimal_swu_per_kg']:.2f}")
            if 'swu_improvement' in self.results['performance_analysis']:
                improvement = self.results['performance_analysis']['swu_improvement']
                report.append(f"  SWU Improvement: {improvement:.2%}")
            report.append("")
        
        # Uncertainty results
        if 'uncertainty' in self.results:
            uq = self.results['uncertainty']['pce_results']
            report.append("UNCERTAINTY QUANTIFICATION:")
            report.append(f"  Mean Product Assay: {uq['mean']:.4f}")
            report.append(f"  Standard Deviation: {uq['standard_deviation']:.4f}")
            ci_low, ci_high = uq['confidence_interval_95']
            report.append(f"  95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
            report.append("")
        
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


# Example usage and testing
def run_example_simulation():
    """Run an example comprehensive simulation."""
    # Configuration for reactor-grade uranium enrichment
    config = {
        'feed_assay': 0.00711,      # Natural uranium
        'product_assay': 0.035,     # Reactor grade
        'tails_assay': 0.0025,      # Typical tails
        'feed_flow_rate': 100.0,    # kg/h
        'stages': 15,
        'machines': 1500,
        'separation_factor': 1.2,
        'simulation_time': 24.0,
        'time_step': 1.0,
        'enable_optimization': True,
        'enable_uncertainty': True,
        'ml_enhancement': True
    }
    
    # Create and run simulation
    sim = IntegratedEnrichmentSimulation(config)
    results = sim.run_comprehensive_simulation()
    
    # Generate and print summary report
    report = sim.generate_summary_report()
    print(report)
    
    # Export results
    export_file = sim.export_results()
    
    return results, export_file


if __name__ == "__main__":
    # Run example simulation
    results, export_file = run_example_simulation()
    print(f"\nSimulation completed! Results exported to: {export_file}")
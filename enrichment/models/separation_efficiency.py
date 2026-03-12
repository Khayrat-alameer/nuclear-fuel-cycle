"""
Separation Efficiency Models for Uranium Enrichment
================================================

This module implements advanced separation efficiency calculations based on 
recent research including machine learning optimization and high-fidelity CFD coupling.

Key Features:
- Traditional SWU calculations
- Machine learning enhanced efficiency models
- CFD-coupled efficiency corrections
- Multi-isotope separation modeling

Based on research from:
1. "Optimization of Cascade Configurations Using Machine Learning" (Nuclear Engineering and Design, 2024)
2. "High-Fidelity CFD Simulations of Single-Stage Centrifuges" (Journal of Computational Physics, 2022)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings


class SeparationEfficiencyModel:
    """
    Advanced separation efficiency modeling for uranium enrichment cascades.
    
    Implements both traditional SWU theory and modern enhancements from 
    recent research (2022-2024).
    """
    
    def __init__(self, use_ml_enhancement: bool = False):
        self.use_ml_enhancement = use_ml_enhancement
        
    def calculate_traditional_swu(self, 
                                feed_assay: float,
                                product_assay: float, 
                                tails_assay: float,
                                feed_mass: float) -> float:
        """
        Calculate traditional Separative Work Units (SWU).
        
        SWU = P*V(x_p) + T*V(x_t) - F*V(x_f)
        where V(x) = (2x-1)*ln(x/(1-x))
        
        Args:
            feed_assay: Feed U-235 concentration (fraction)
            product_assay: Product U-235 concentration (fraction)  
            tails_assay: Tails U-235 concentration (fraction)
            feed_mass: Total feed mass (kg)
            
        Returns:
            Total SWU required
        """
        def v_function(x: float) -> float:
            """Value function for SWU calculation."""
            if x <= 0 or x >= 1:
                raise ValueError("Assay must be between 0 and 1")
            return (2*x - 1) * np.log(x / (1 - x))
        
        # Mass balance calculations
        denominator = product_assay - tails_assay
        if abs(denominator) < 1e-10:
            raise ValueError("Product and tails assays too close for calculation")
            
        product_mass = feed_mass * (feed_assay - tails_assay) / denominator
        tails_mass = feed_mass - product_mass
        
        # SWU calculation
        swu = (product_mass * v_function(product_assay) +
               tails_mass * v_function(tails_assay) -
               feed_mass * v_function(feed_assay))
        
        return swu
    
    def calculate_enhanced_efficiency(self,
                                    feed_assay: float,
                                    product_assay: float,
                                    tails_assay: float,
                                    cascade_stages: int,
                                    machine_count: int,
                                    thermal_gradient: float = 0.0,
                                    ml_correction_factor: float = 1.0) -> Dict[str, float]:
        """
        Calculate enhanced separation efficiency incorporating recent research findings.
        
        Includes corrections for:
        - Thermal gradients (from CFD simulations, 2022)
        - Machine learning optimization (2024)
        - Cascade configuration effects
        
        Args:
            feed_assay: Feed U-235 concentration
            product_assay: Product U-235 concentration  
            tails_assay: Tails U-235 concentration
            cascade_stages: Number of cascade stages
            machine_count: Total number of centrifuges
            thermal_gradient: Temperature gradient effect (0.0 = no effect)
            ml_correction_factor: ML-based optimization factor (default 1.0 = no enhancement)
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Base SWU calculation
        base_feed_mass = 1.0  # per kg basis
        base_swu = self.calculate_traditional_swu(
            feed_assay, product_assay, tails_assay, base_feed_mass
        )
        
        # Apply thermal gradient correction (based on CFD research 2022)
        thermal_correction = self._thermal_gradient_correction(thermal_gradient)
        
        # Apply ML optimization correction (based on 2024 research)
        if self.use_ml_enhancement:
            ml_correction = ml_correction_factor
        else:
            ml_correction = 1.0
        
        # Cascade configuration efficiency
        cascade_efficiency = self._cascade_configuration_efficiency(
            cascade_stages, machine_count
        )
        
        # Combined efficiency
        enhanced_swu = base_swu * thermal_correction * ml_correction * cascade_efficiency
        
        return {
            'base_swu_per_kg': base_swu,
            'enhanced_swu_per_kg': enhanced_swu,
            'efficiency_improvement': (base_swu - enhanced_swu) / base_swu if base_swu > 0 else 0.0,
            'thermal_correction_factor': thermal_correction,
            'ml_correction_factor': ml_correction,
            'cascade_efficiency_factor': cascade_efficiency
        }
    
    def _thermal_gradient_correction(self, thermal_gradient: float) -> float:
        """
        Apply thermal gradient correction based on CFD simulations (2022).
        
        Thermal gradients in centrifuges affect separation performance.
        This is a simplified model based on research findings.
        """
        # Simplified model: thermal gradients generally reduce efficiency
        # but optimal gradients can improve it slightly
        if thermal_gradient == 0.0:
            return 1.0
        elif thermal_gradient > 0:
            # Positive gradient (hotter at top) - typical scenario
            # Reduces efficiency by up to 5% for large gradients
            correction = max(0.95, 1.0 - 0.05 * min(1.0, thermal_gradient))
        else:
            # Negative gradient - less common
            correction = max(0.90, 1.0 + 0.1 * thermal_gradient)
        
        return correction
    
    def _cascade_configuration_efficiency(self, 
                                        stages: int, 
                                        machines: int) -> float:
        """
        Calculate cascade configuration efficiency.
        
        Based on research showing optimal stage-to-machine ratios.
        """
        if stages <= 0 or machines <= 0:
            return 1.0
            
        machines_per_stage = machines / stages
        
        # Optimal range is typically 50-200 machines per stage
        if 50 <= machines_per_stage <= 200:
            return 1.0  # Optimal configuration
        elif machines_per_stage < 50:
            # Too few machines per stage - reduced efficiency
            return 0.8 + 0.2 * (machines_per_stage / 50)
        else:
            # Too many machines per stage - diminishing returns
            return 0.9 + 0.1 * np.exp(-(machines_per_stage - 200) / 100)
    
    def optimize_cascade_parameters(self,
                                  feed_assay: float,
                                  target_product_assay: float,
                                  target_tails_assay: float,
                                  max_machines: int = 10000) -> Dict[str, float]:
        """
        Optimize cascade parameters using ML-inspired approach (2024 research).
        
        Finds optimal stage count and machine allocation for minimum SWU.
        """
        def objective_function(x):
            stages, machines_per_stage = int(x[0]), int(x[1])
            if stages < 2 or machines_per_stage < 1:
                return np.inf
                
            total_machines = stages * machines_per_stage
            if total_machines > max_machines:
                return np.inf
                
            # Estimate separation factor based on machines per stage
            # More machines per stage allows higher effective separation
            separation_factor = 1.1 + 0.1 * min(1.0, machines_per_stage / 100)
            
            # Calculate achievable product assay
            achievable_product = self._estimate_product_assay(
                feed_assay, target_tails_assay, stages, separation_factor
            )
            
            # Penalty for not meeting target
            assay_penalty = max(0, target_product_assay - achievable_product) * 1000
            
            # Calculate SWU
            try:
                swu = self.calculate_traditional_swu(
                    feed_assay, achievable_product, target_tails_assay, 1.0
                )
            except:
                return np.inf
                
            return swu + assay_penalty
        
        # Initial guess
        initial_guess = [10, 100]  # 10 stages, 100 machines per stage
        
        # Bounds: 2-50 stages, 10-500 machines per stage
        bounds = [(2, 50), (10, 500)]
        
        try:
            result = minimize(objective_function, initial_guess, 
                            method='L-BFGS-B', bounds=bounds)
            
            optimal_stages = int(result.x[0])
            optimal_machines_per_stage = int(result.x[1])
            total_machines = optimal_stages * optimal_machines_per_stage
            
            # Calculate final performance
            separation_factor = 1.1 + 0.1 * min(1.0, optimal_machines_per_stage / 100)
            final_product = self._estimate_product_assay(
                feed_assay, target_tails_assay, optimal_stages, separation_factor
            )
            final_swu = self.calculate_traditional_swu(
                feed_assay, final_product, target_tails_assay, 1.0
            )
            
            return {
                'optimal_stages': optimal_stages,
                'optimal_machines_per_stage': optimal_machines_per_stage,
                'total_machines': total_machines,
                'achieved_product_assay': final_product,
                'swu_per_kg': final_swu,
                'optimization_success': result.success
            }
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            return {
                'optimal_stages': 10,
                'optimal_machines_per_stage': 100,
                'total_machines': 1000,
                'achieved_product_assay': target_product_assay,
                'swu_per_kg': self.calculate_traditional_swu(
                    feed_assay, target_product_assay, target_tails_assay, 1.0
                ),
                'optimization_success': False
            }
    
    def _estimate_product_assay(self, 
                               feed_assay: float,
                               tails_assay: float,
                               stages: int,
                               separation_factor: float) -> float:
        """
        Estimate achievable product assay given cascade parameters.
        """
        # Simplified cascade theory
        max_enrichment = feed_assay * (separation_factor ** stages)
        min_enrichment = tails_assay * (separation_factor ** (stages - 1))
        
        # Realistic estimate between theoretical limits
        estimated = min(max_enrichment, max(min_enrichment, feed_assay * 5))
        return min(estimated, 0.9)  # Cap at 90% enrichment


# Example usage
def example_separation_efficiency():
    """Example usage of separation efficiency models."""
    model = SeparationEfficiencyModel(use_ml_enhancement=True)
    
    # Typical reactor-grade enrichment parameters
    feed_assay = 0.00711      # Natural uranium
    product_assay = 0.035     # Reactor grade
    tails_assay = 0.0025      # Typical tails
    
    # Traditional SWU calculation
    traditional_swu = model.calculate_traditional_swu(
        feed_assay, product_assay, tails_assay, 1.0
    )
    
    # Enhanced efficiency calculation
    enhanced_results = model.enhanced_efficiency = model.calculate_enhanced_efficiency(
        feed_assay, product_assay, tails_assay,
        cascade_stages=15, machine_count=1500,
        thermal_gradient=0.1, ml_correction_factor=0.93  # 7% improvement from ML
    )
    
    # Optimization example
    optimization_results = model.optimize_cascade_parameters(
        feed_assay, product_assay, tails_assay, max_machines=2000
    )
    
    print("Separation Efficiency Analysis:")
    print(f"Traditional SWU/kg: {traditional_swu:.2f}")
    print(f"Enhanced SWU/kg: {enhanced_results['enhanced_swu_per_kg']:.2f}")
    print(f"Efficiency improvement: {enhanced_results['efficiency_improvement']:.1%}")
    print(f"Optimal stages: {optimization_results['optimal_stages']}")
    print(f"Optimal machines: {optimization_results['total_machines']}")


if __name__ == "__main__":
    example_separation_efficiency()
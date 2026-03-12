"""
Uncertainty Quantification for Uranium Mining Simulations
===============================================

This module implements uncertainty quantification methods for uranium mining 
simulations, focusing on resource estimation uncertainty, extraction efficiency 
variability, and environmental impact prediction confidence intervals.

Key Features:
- Monte Carlo simulation for parameter uncertainty propagation
- Polynomial Chaos Expansion for efficient uncertainty quantification
- Sensitivity analysis using Sobol indices
- Bayesian updating for data assimilation
- Confidence intervals for key performance metrics

Based on integrated approaches from recent mining and nuclear engineering research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm, lognorm, uniform
from scipy.integrate import solve_ivp


class MiningUncertaintyQuantification:
    """
    Comprehensive uncertainty quantification framework for uranium mining simulations.
    
    Implements multiple uncertainty quantification methods suitable for mining applications.
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 confidence_level: float = 0.95):
        self.num_samples = num_samples
        self.confidence_level = confidence_level
        self.uncertain_parameters = {}
        self.results = {}
        
    def define_uncertain_parameters(self,
                                parameters: Dict[str, Dict[str, float]]) -> None:
        """
        Define uncertain parameters with their probability distributions.
        
        Args:
            parameters: Dictionary with parameter names as keys and distribution 
                       parameters as values. Supported distributions:
                       - 'normal': {'mean': float, 'std': float}
                       - 'lognormal': {'mean': float, 'std': float}  
                       - 'uniform': {'min': float, 'max': float}
        """
        self.uncertain_parameters = parameters
        
    def monte_carlo_simulation(self,
                             simulation_function,
                             output_function) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo simulation for uncertainty propagation.
        
        Args:
            simulation_function: Function that runs the mining simulation
            output_function: Function that extracts output of interest
            
        Returns:
            Dictionary with Monte Carlo results and statistics
        """
        mc_outputs = []
        valid_samples = 0
        
        for i in range(self.num_samples):
            # Sample uncertain parameters
            sampled_params = {}
            for param_name, param_info in self.uncertain_parameters.items():
                dist_type = param_info.get('distribution', 'normal')
                
                if dist_type == 'normal':
                    sampled_params[param_name] = np.random.normal(
                        param_info['mean'], param_info['std']
                    )
                elif dist_type == 'lognormal':
                    # Convert to lognormal parameters
                    mean = param_info['mean']
                    std = param_info['std']
                    sigma = np.sqrt(np.log(1 + (std/mean)**2))
                    mu = np.log(mean) - 0.5 * sigma**2
                    sampled_params[param_name] = np.random.lognormal(mu, sigma)
                elif dist_type == 'uniform':
                    sampled_params[param_name] = np.random.uniform(
                        param_info['min'], param_info['max']
                    )
                else:
                    raise ValueError(f"Unsupported distribution: {dist_type}")
            
            try:
                # Run simulation
                simulation_result = simulation_function(sampled_params)
                output_value = output_function(simulation_result)
                mc_outputs.append(output_value)
                valid_samples += 1
            except Exception as e:
                # Skip failed simulations
                continue
        
        if valid_samples == 0:
            raise ValueError("All Monte Carlo simulations failed.")
        
        mc_outputs = np.array(mc_outputs)
        
        # Calculate statistics
        mean_output = np.mean(mc_outputs)
        std_output = np.std(mc_outputs)
        median_output = np.median(mc_outputs)
        
        # Calculate confidence intervals
        alpha = 1.0 - self.confidence_level
        lower_percentile = (alpha / 2.0) * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100
        
        confidence_interval = (
            np.percentile(mc_outputs, lower_percentile),
            np.percentile(mc_outputs, upper_percentile)
        )
        
        self.results['monte_carlo'] = {
            'samples': mc_outputs,
            'mean': mean_output,
            'std': std_output,
            'median': median_output,
            'confidence_interval': confidence_interval,
            'valid_samples': valid_samples,
            'total_samples': self.num_samples
        }
        
        return self.results['monte_carlo']
    
    def polynomial_chaos_expansion(self,
                                simulation_function,
                                output_function,
                                polynomial_order: int = 3) -> Dict[str, any]:
        """
        Implement Polynomial Chaos Expansion for efficient uncertainty quantification.
        
        Note: This is a simplified implementation. Full PCE would require more sophisticated
        orthogonal polynomial basis and quadrature rules.
        """
        # Generate collocation points using Latin Hypercube Sampling
        n_params = len(self.uncertain_parameters)
        collocation_points = self._generate_latin_hypercube_samples(
            n_params, polynomial_order + 1
        )
        
        # Evaluate model at collocation points
        model_outputs = []
        for point in collocation_points:
            # Map collocation point to physical parameters
            params = self._map_to_physical_parameters(point)
            
            try:
                simulation_result = simulation_function(params)
                output_value = output_function(simulation_result)
                model_outputs.append(output_value)
            except Exception as e:
                model_outputs.append(np.nan)
        
        model_outputs = np.array(model_outputs)
        
        # Remove NaN values
        valid_mask = ~np.isnan(model_outputs)
        if np.sum(valid_mask) == 0:
            raise ValueError("All PCE evaluations failed.")
        
        valid_outputs = model_outputs[valid_mask]
        valid_points = collocation_points[valid_mask]
        
        # Calculate PCE coefficients (simplified linear regression)
        # In practice, would use orthogonal projection or regression
        if len(valid_outputs) < len(valid_points[0]):
            # Not enough samples for regression
            mean_output = np.mean(valid_outputs)
            std_output = np.std(valid_outputs)
        else:
            # Simple linear regression approximation
            X = np.column_stack([np.ones(len(valid_points)), valid_points])
            coeffs = np.linalg.lstsq(X, valid_outputs, rcond=None)[0]
            mean_output = coeffs[0]
            std_output = np.std(valid_outputs - X @ coeffs)
        
        # Calculate confidence interval
        alpha = 1.0 - self.confidence_level
        lower_bound = mean_output - norm.ppf(1 - alpha/2) * std_output
        upper_bound = mean_output + norm.ppf(1 - alpha/2) * std_output
        
        self.results['polynomial_chaos'] = {
            'mean': mean_output,
            'std': std_output,
            'confidence_interval': (lower_bound, upper_bound),
            'collocation_points': valid_points,
            'model_outputs': valid_outputs
        }
        
        return self.results['polynomial_chaos']
    
    def _generate_latin_hypercube_samples(self, 
                                       n_dimensions: int, 
                                       n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n_samples, n_dimensions))
        
        for i in range(n_dimensions):
            # Create equally spaced intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            # Randomly sample within each interval
            random_offsets = np.random.uniform(0, 1/n_samples, n_samples)
            samples[:, i] = intervals[:-1] + random_offsets
            # Shuffle to ensure randomness across dimensions
            np.random.shuffle(samples[:, i])
        
        return samples
    
    def _map_to_physical_parameters(self, 
                                 normalized_point: np.ndarray) -> Dict[str, float]:
        """Map normalized point [0,1] to physical parameter values."""
        physical_params = {}
        
        for i, (param_name, param_info) in enumerate(self.uncertain_parameters.items()):
            normalized_value = normalized_point[i]
            dist_type = param_info.get('distribution', 'uniform')
            
            if dist_type == 'normal':
                # Use inverse CDF of normal distribution
                physical_params[param_name] = norm.ppf(
                    normalized_value, 
                    loc=param_info['mean'], 
                    scale=param_info['std']
                )
            elif dist_type == 'lognormal':
                mean = param_info['mean']
                std = param_info['std']
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                mu = np.log(mean) - 0.5 * sigma**2
                physical_params[param_name] = lognorm.ppf(
                    normalized_value, s=sigma, scale=np.exp(mu)
                )
            elif dist_type == 'uniform':
                physical_params[param_name] = (param_info['min'] + 
                                            normalized_value * (param_info['max'] - param_info['min']))
        
        return physical_params
    
    def sensitivity_analysis(self,
                         simulation_function,
                         output_function,
                         method: str = 'sobol') -> Dict[str, float]:
        """
        Perform sensitivity analysis to identify influential parameters.
        
        Args:
            simulation_function: Function that runs the mining simulation
            output_function: Function that extracts output of interest
            method: Sensitivity analysis method ('sobol', 'morris', 'regression')
            
        Returns:
            Dictionary with sensitivity indices for each parameter
        """
        if method == 'sobol':
            return self._sobol_sensitivity_analysis(simulation_function, output_function)
        elif method == 'morris':
            return self._morris_sensitivity_analysis(simulation_function, output_function)
        elif method == 'regression':
            return self._regression_sensitivity_analysis(simulation_function, output_function)
        else:
            raise ValueError(f"Unknown sensitivity analysis method: {method}")
    
    def _sobol_sensitivity_analysis(self,
                                 simulation_function,
                                 output_function) -> Dict[str, float]:
        """Perform Sobol sensitivity analysis (first-order indices)."""
        # Generate Sobol sequences
        n_params = len(self.uncertain_parameters)
        n_samples = max(100, self.num_samples // 10)
        
        # Generate two independent samples
        A = np.random.uniform(0, 1, (n_samples, n_params))
        B = np.random.uniform(0, 1, (n_samples, n_params))
        
        # Calculate f(A) and f(B)
        f_A = []
        f_B = []
        
        for i in range(n_samples):
            params_A = self._map_to_physical_parameters(A[i])
            params_B = self._map_to_physical_parameters(B[i])
            
            try:
                f_A.append(output_function(simulation_function(params_A)))
                f_B.append(output_function(simulation_function(params_B)))
            except:
                f_A.append(0.0)
                f_B.append(0.0)
        
        f_A = np.array(f_A)
        f_B = np.array(f_B)
        
        # Calculate first-order Sobol indices
        sensitivity_indices = {}
        f0_sq = np.var(f_B)  # Estimate of V(Y)
        
        if f0_sq == 0:
            # No variance, all parameters have zero sensitivity
            for param_name in self.uncertain_parameters.keys():
                sensitivity_indices[param_name] = 0.0
            return sensitivity_indices
        
        for i, param_name in enumerate(self.uncertain_parameters.keys()):
            # Create matrix C where only i-th column is from A, others from B
            C = B.copy()
            C[:, i] = A[:, i]
            
            f_C = []
            for j in range(n_samples):
                params_C = self._map_to_physical_parameters(C[j])
                try:
                    f_C.append(output_function(simulation_function(params_C)))
                except:
                    f_C.append(0.0)
            
            f_C = np.array(f_C)
            
            # First-order index: S_i = Cov(f(A), f(C)) / Var(f(B))
            cov_fA_fC = np.cov(f_A, f_C)[0, 1]
            S_i = cov_fA_fC / f0_sq if f0_sq > 0 else 0.0
            sensitivity_indices[param_name] = max(0.0, min(1.0, S_i))
        
        return sensitivity_indices
    
    def _morris_sensitivity_analysis(self,
                                  simulation_function,
                                  output_function) -> Dict[str, float]:
        """Perform Morris elementary effects sensitivity analysis."""
        n_params = len(self.uncertain_parameters)
        n_trajectories = max(10, self.num_samples // 20)
        
        sensitivity_indices = {param_name: [] for param_name in self.uncertain_parameters.keys()}
        
        for trajectory in range(n_trajectories):
            # Random starting point
            x_base = np.random.uniform(0, 1, n_params)
            
            # Random order of parameters
            param_order = np.random.permutation(n_params)
            
            # Calculate elementary effects
            y_current = None
            for step, param_idx in enumerate(param_order):
                x_current = x_base.copy()
                
                # Move in direction of parameter
                delta = 0.1  # Step size
                x_current[param_idx] = min(1.0, x_current[param_idx] + delta)
                
                params = self._map_to_physical_parameters(x_current)
                try:
                    y_new = output_function(simulation_function(params))
                except:
                    y_new = 0.0
                
                if y_current is not None:
                    elementary_effect = (y_new - y_current) / delta
                    param_name = list(self.uncertain_parameters.keys())[param_idx]
                    sensitivity_indices[param_name].append(abs(elementary_effect))
                
                y_current = y_new
                x_base = x_current
        
        # Calculate mean elementary effects
        morris_indices = {}
        for param_name, effects in sensitivity_indices.items():
            if effects:
                morris_indices[param_name] = np.mean(effects)
            else:
                morris_indices[param_name] = 0.0
        
        return morris_indices
    
    def _regression_sensitivity_analysis(self,
                                     simulation_function,
                                     output_function) -> Dict[str, float]:
        """Perform regression-based sensitivity analysis."""
        n_samples = max(100, self.num_samples // 5)
        
        # Generate samples and evaluate model
        X = []
        Y = []
        
        for i in range(n_samples):
            normalized_point = np.random.uniform(0, 1, len(self.uncertain_parameters))
            params = self._map_to_physical_parameters(normalized_point)
            
            try:
                y = output_function(simulation_function(params))
                X.append(normalized_point)
                Y.append(y)
            except:
                continue
        
        if len(Y) < 10:
            # Not enough successful evaluations
            return {param_name: 0.0 for param_name in self.uncertain_parameters.keys()}
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Standardize variables
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y_std = (Y - np.mean(Y)) / np.std(Y)
        
        # Linear regression
        coeffs = np.linalg.lstsq(X_std, Y_std, rcond=None)[0]
        
        # R-squared values as sensitivity indices
        regression_indices = {}
        for i, param_name in enumerate(self.uncertain_parameters.keys()):
            regression_indices[param_name] = coeffs[i] ** 2
        
        return regression_indices
    
    def bayesian_updating(self,
                      prior_parameters: Dict[str, Dict[str, float]],
                      observed_data: Dict[str, float],
                      likelihood_function) -> Dict[str, Dict[str, float]]:
        """
        Perform Bayesian updating of parameter distributions using observed data.
        
        Args:
            prior_parameters: Prior parameter distributions
            observed_data: Observed data for updating
            likelihood_function: Function that calculates likelihood of data given parameters
            
        Returns:
            Updated (posterior) parameter distributions
        """
        # This is a simplified implementation
        # Full Bayesian updating would require MCMC or other advanced methods
        
        posterior_parameters = {}
        
        for param_name, prior_info in prior_parameters.items():
            # Simple conjugate update for normal distributions
            if prior_info.get('distribution') == 'normal':
                prior_mean = prior_info['mean']
                prior_std = prior_info['std']
                
                # Assume observed data provides sample mean and std
                if param_name in observed_data:
                    observed_value = observed_data[param_name]
                    # Simple weighted average update
                    weight_prior = 1.0 / (prior_std ** 2)
                    weight_observed = 1.0 / (prior_std ** 2)  # Assume same uncertainty
                    
                    posterior_mean = (weight_prior * prior_mean + weight_observed * observed_value) / (weight_prior + weight_observed)
                    posterior_std = np.sqrt(1.0 / (weight_prior + weight_observed))
                    
                    posterior_parameters[param_name] = {
                        'distribution': 'normal',
                        'mean': posterior_mean,
                        'std': posterior_std
                    }
                else:
                    posterior_parameters[param_name] = prior_info
            else:
                # For non-normal distributions, keep prior unchanged
                posterior_parameters[param_name] = prior_info
        
        return posterior_parameters


# Example usage functions
def example_mining_simulation(params: Dict[str, float]) -> Dict[str, float]:
    """
    Example mining simulation function for uncertainty quantification.
    
    This is a simplified example that demonstrates the interface.
    """
    # Extract parameters
    ore_grade = params.get('ore_grade', 0.2)
    recovery_rate = params.get('recovery_rate', 0.85)
    mining_cost = params.get('mining_cost', 15.0)
    uranium_price = params.get('uranium_price', 100.0)
    
    # Simple economic calculation
    revenue_per_tonne = ore_grade * recovery_rate * uranium_price * 2204.62  # lb to tonnes
    profit_per_tonne = revenue_per_tonne - mining_cost
    
    return {
        'profit_per_tonne': profit_per_tonne,
        'revenue_per_tonne': revenue_per_tonne,
        'ore_grade': ore_grade,
        'recovery_rate': recovery_rate
    }


def example_uncertainty_quantification():
    """Example usage of uncertainty quantification framework."""
    # Initialize UQ model
    uq_model = MiningUncertaintyQuantification(num_samples=1000, confidence_level=0.95)
    
    # Define uncertain parameters
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
    
    uq_model.define_uncertain_parameters(uncertain_params)
    
    # Define output function
    def extract_profit(simulation_result: Dict[str, float]) -> float:
        return simulation_result['profit_per_tonne']
    
    # Monte Carlo simulation
    mc_results = uq_model.monte_carlo_simulation(example_mining_simulation, extract_profit)
    
    # Polynomial Chaos Expansion (simplified)
    pce_results = uq_model.polynomial_chaos_expansion(example_mining_simulation, extract_profit)
    
    # Sensitivity analysis
    sobol_results = uq_model.sensitivity_analysis(example_mining_simulation, extract_profit, method='sobol')
    morris_results = uq_model.sensitivity_analysis(example_mining_simulation, extract_profit, method='morris')
    
    print("Uncertainty Quantification Results:")
    print(f"Monte Carlo Mean Profit: ${mc_results['mean']:.2f}/tonne")
    print(f"Monte Carlo Std Dev: ${mc_results['std']:.2f}/tonne")
    print(f"95% Confidence Interval: [${mc_results['confidence_interval'][0]:.2f}, ${mc_results['confidence_interval'][1]:.2f}]")
    print()
    
    print("Polynomial Chaos Results:")
    print(f"PCE Mean Profit: ${pce_results['mean']:.2f}/tonne")
    print(f"PCE Std Dev: ${pce_results['std']:.2f}/tonne")
    print(f"PCE 95% CI: [${pce_results['confidence_interval'][0]:.2f}, ${pce_results['confidence_interval'][1]:.2f}]")
    print()
    
    print("Sensitivity Analysis (Sobol):")
    for param, index in sobol_results.items():
        print(f"  {param}: {index:.3f}")
    print()
    
    print("Sensitivity Analysis (Morris):")
    for param, index in morris_results.items():
        print(f"  {param}: {index:.3f}")
    
    return mc_results, pce_results, sobol_results, morris_results


if __name__ == "__main__":
    example_uncertainty_quantification()
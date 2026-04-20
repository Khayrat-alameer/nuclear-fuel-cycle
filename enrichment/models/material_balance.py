"""
Material Balance Equations for Uranium Enrichment Cascades
=======================================================

This module implements comprehensive material balance equations for gas centrifuge 
cascades based on recent research (2023-2024). Includes time-dependent modeling,
multi-isotope tracking, and countercurrent flow dynamics.

Based on research from:
1. "Dynamic Modeling and Simulation of Gas Centrifuge Cascades" (Annals of Nuclear Energy, 2023)
2. "A Modular Simulation Framework for Multi-Isotope Centrifuge Cascades" (arXiv:2405.11289, 2024)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class IsotopeComposition:
    """Represents isotopic composition of uranium."""
    u234: float = 0.000054  # Natural abundance
    u235: float = 0.00711   # Natural abundance  
    u238: float = 0.992836  # Natural abundance
    
    def normalize(self):
        """Normalize composition to sum to 1.0."""
        total = self.u234 + self.u235 + self.u238
        if total > 0:
            self.u234 /= total
            self.u235 /= total
            self.u238 /= total
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.u234, self.u235, self.u238])
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create from numpy array."""
        return cls(u234=arr[0], u235=arr[1], u238=arr[2])


class MaterialBalanceModel:
    """
    Comprehensive material balance model for enrichment cascades.
    
    Implements time-dependent material balance equations with multi-isotope tracking
    and countercurrent flow dynamics as described in recent literature.
    """
    
    def __init__(self, 
                 stages: int,
                 time_step: float = 1.0,
                 simulation_time: float = 24.0):
        self.stages = stages
        self.time_step = time_step
        self.simulation_time = simulation_time
        self.time_points = np.arange(0, simulation_time + time_step, time_step)
        
        # Initialize state arrays
        self.isotopes = ['U234', 'U235', 'U238']
        self.compositions = np.zeros((len(self.time_points), stages, 3))
        self.flow_rates = np.zeros((len(self.time_points), stages))
        self.pressures = np.zeros((len(self.time_points), stages))
        
        # Feed, product, and tails streams
        self.feed_composition = IsotopeComposition()
        self.product_streams = np.zeros((len(self.time_points), 3))
        self.tails_streams = np.zeros((len(self.time_points), 3))
        
    def set_initial_conditions(self,
                             feed_composition: Optional[IsotopeComposition] = None,
                             initial_flow_rate: float = 100.0,
                             initial_pressure: float = 1.0):
        """Set initial conditions for the cascade."""
        if feed_composition is not None:
            self.feed_composition = feed_composition
        
        # Initialize compositions with feed composition
        feed_arr = self.feed_composition.to_array()
        for t in range(len(self.time_points)):
            for stage in range(self.stages):
                self.compositions[t, stage, :] = feed_arr
        
        # Initialize flow rates and pressures
        self.flow_rates[0, :] = initial_flow_rate / self.stages
        self.pressures[0, :] = initial_pressure
        
    def apply_separation_matrix(self, 
                              separation_factors: Dict[str, float],
                              stage: int,
                              time_idx: int) -> np.ndarray:
        """
        Apply separation matrix to a stage's composition.
        
        Args:
            separation_factors: Dictionary with separation factors for each isotope
            stage: Stage index
            time_idx: Time index
            
        Returns:
            Separated composition array
        """
        current_comp = self.compositions[time_idx, stage, :]
        
        # Apply separation factors
        separated = np.zeros(3)
        total_weighted = 0.0
        
        for i, isotope in enumerate(self.isotopes):
            if isotope == 'U235':
                sf = separation_factors.get('U235', 1.2)
            elif isotope == 'U234':
                sf = separation_factors.get('U234', 1.005)  # U234 follows U235
            else:  # U238
                sf = separation_factors.get('U238', 1.0)
            
            separated[i] = current_comp[i] * sf
            total_weighted += separated[i]
        
        if total_weighted > 0:
            separated /= total_weighted
        
        return separated
    
    def solve_material_balance(self,
                             feed_flow_rate: float,
                             product_withdrawal_stage: int = -1,
                             tails_withdrawal_stage: int = 0,
                             separation_factors: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Solve time-dependent material balance equations.
        
        Implements countercurrent flow with feed input, product withdrawal,
        and tails withdrawal as described in dynamic cascade modeling research.
        """
        if separation_factors is None:
            separation_factors = {'U235': 1.2, 'U234': 1.005, 'U238': 1.0}
        
        # Set product and tails withdrawal stages
        if product_withdrawal_stage < 0:
            product_withdrawal_stage = self.stages - 1
        
        # Main simulation loop
        for t_idx in range(1, len(self.time_points)):
            # Handle feed input (typically at optimal stage)
            feed_stage = self._determine_feed_stage(t_idx)
            self._add_feed(feed_flow_rate, feed_stage, t_idx)
            
            # Handle countercurrent flow between stages
            self._update_countercurrent_flow(t_idx)
            
            # Apply separation in each stage
            for stage in range(self.stages):
                separated_comp = self.apply_separation_matrix(
                    separation_factors, stage, t_idx - 1
                )
                self.compositions[t_idx, stage, :] = separated_comp
            
            # Handle product and tails withdrawal
            self._withdraw_product(product_withdrawal_stage, t_idx)
            self._withdraw_tails(tails_withdrawal_stage, t_idx)
            
            # Ensure mass conservation
            self._enforce_mass_conservation(t_idx)
        
        return {
            'time': self.time_points,
            'compositions': self.compositions,
            'flow_rates': self.flow_rates,
            'product_streams': self.product_streams,
            'tails_streams': self.tails_streams
        }
    
    def _determine_feed_stage(self, t_idx: int) -> int:
        """Determine optimal feed stage based on current cascade state."""
        # Simplified: feed at middle stage initially, adjust based on composition
        if t_idx == 1:
            return self.stages // 2
        else:
            # Find stage closest to feed composition
            feed_arr = self.feed_composition.to_array()
            distances = np.sum((self.compositions[t_idx-1, :, :] - feed_arr) ** 2, axis=1)
            return int(np.argmin(distances))
    
    def _add_feed(self, feed_flow_rate: float, feed_stage: int, t_idx: int):
        """Add feed stream to the specified stage."""
        # Distribute feed flow across time step
        dt_feed = feed_flow_rate * self.time_step
        
        # Update flow rate at feed stage
        self.flow_rates[t_idx, feed_stage] += dt_feed
        
        # Update composition with mass-weighted average
        current_mass = self.flow_rates[t_idx-1, feed_stage] * self.time_step
        new_mass = current_mass + dt_feed
        
        if new_mass > 0:
            current_comp = self.compositions[t_idx-1, feed_stage, :]
            feed_comp = self.feed_composition.to_array()
            new_comp = (current_comp * current_mass + feed_comp * dt_feed) / new_mass
            self.compositions[t_idx, feed_stage, :] = new_comp
    
    def _update_countercurrent_flow(self, t_idx: int):
        """Update flow between stages based on countercurrent principle."""
        # Simplified countercurrent flow model
        for stage in range(self.stages):
            if stage == 0:
                # First stage: only receives from stage 1
                if self.stages > 1:
                    self.flow_rates[t_idx, stage] = self.flow_rates[t_idx-1, stage+1]
                    self.compositions[t_idx, stage, :] = self.compositions[t_idx-1, stage+1, :]
            elif stage == self.stages - 1:
                # Last stage: only receives from stage -2
                self.flow_rates[t_idx, stage] = self.flow_rates[t_idx-1, stage-1]
                self.compositions[t_idx, stage, :] = self.compositions[t_idx-1, stage-1, :]
            else:
                # Internal stages: average of neighbors
                avg_flow = (self.flow_rates[t_idx-1, stage-1] + self.flow_rates[t_idx-1, stage+1]) / 2
                avg_comp = (self.compositions[t_idx-1, stage-1, :] + self.compositions[t_idx-1, stage+1, :]) / 2
                self.flow_rates[t_idx, stage] = avg_flow
                self.compositions[t_idx, stage, :] = avg_comp
    
    def _withdraw_product(self, product_stage: int, t_idx: int):
        """Withdraw product stream from specified stage."""
        # Withdraw a fraction of the flow as product
        product_fraction = 0.1  # 10% of stage flow as product
        product_flow = self.flow_rates[t_idx, product_stage] * product_fraction
        
        # Store product composition
        self.product_streams[t_idx, :] = self.compositions[t_idx, product_stage, :]
        
        # Reduce flow rate at product stage
        self.flow_rates[t_idx, product_stage] *= (1 - product_fraction)
    
    def _withdraw_tails(self, tails_stage: int, t_idx: int):
        """Withdraw tails stream from specified stage."""
        # Withdraw a fraction of the flow as tails
        tails_fraction = 0.1  # 10% of stage flow as tails
        tails_flow = self.flow_rates[t_idx, tails_stage] * tails_fraction
        
        # Store tails composition
        self.tails_streams[t_idx, :] = self.compositions[t_idx, tails_stage, :]
        
        # Reduce flow rate at tails stage
        self.flow_rates[t_idx, tails_stage] *= (1 - tails_fraction)
    
    def _enforce_mass_conservation(self, t_idx: int):
        """Enforce mass conservation constraints."""
        # Ensure compositions sum to 1.0
        for stage in range(self.stages):
            comp_sum = np.sum(self.compositions[t_idx, stage, :])
            if comp_sum > 0:
                self.compositions[t_idx, stage, :] /= comp_sum
        
        # Ensure non-negative flow rates
        self.flow_rates[t_idx, :] = np.maximum(self.flow_rates[t_idx, :], 0.0)
    
    def calculate_mass_balance_error(self) -> float:
        """Calculate mass balance error over entire simulation."""
        initial_mass = np.sum(self.flow_rates[0, :]) * self.time_step
        final_mass = np.sum(self.flow_rates[-1, :]) * self.time_step
        
        total_product = np.sum(self.product_streams) * self.time_step
        total_tails = np.sum(self.tails_streams) * self.time_step
        total_feed = np.sum(self.flow_rates[:, self._determine_feed_stage(1)]) * self.time_step
        
        # Mass balance: Feed = Product + Tails + Accumulation
        accumulation = final_mass - initial_mass
        balance_error = abs(total_feed - total_product - total_tails - accumulation)
        
        return balance_error / total_feed if total_feed > 0 else 0.0


# Example usage
def example_material_balance():
    """Example usage of material balance model."""
    # Create model with 10 stages
    model = MaterialBalanceModel(stages=10, time_step=1.0, simulation_time=24.0)
    
    # Set initial conditions
    feed_comp = IsotopeComposition(u234=0.000054, u235=0.00711, u238=0.992836)
    model.set_initial_conditions(feed_composition=feed_comp, initial_flow_rate=100.0)
    
    # Solve material balance
    results = model.solve_material_balance(
        feed_flow_rate=100.0,
        product_withdrawal_stage=9,
        tails_withdrawal_stage=0,
        separation_factors={'U235': 1.2, 'U234': 1.005, 'U238': 1.0}
    )
    
    # Calculate mass balance error
    error = model.calculate_mass_balance_error()
    
    print("Material Balance Results:")
    print(f"Final product U235 concentration: {results['product_streams'][-1, 1]:.4f}")
    print(f"Final tails U235 concentration: {results['tails_streams'][-1, 1]:.4f}")
    print(f"Mass balance error: {error:.2%}")
    
    return results


if __name__ == "__main__":
    example_material_balance()
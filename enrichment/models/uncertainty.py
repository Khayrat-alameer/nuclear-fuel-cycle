"""
Uncertainty Quantification for Uranium Enrichment Cascade Simulations
================================================================

Implements uncertainty quantification using Polynomial Chaos Expansion (PCE)
and Monte Carlo methods for enrichment cascade simulations.

Key Features:
- Polynomial Chaos Expansion (PCE) via regression on Hermite basis
- Monte Carlo validation with confidence intervals
- Sobol sensitivity indices from PCE decomposition

Based on:
"Uncertainty Quantification in Enrichment Cascade Simulations Using
Polynomial Chaos Expansion" (Reliability Engineering & System Safety, 2025)
"""

import numpy as np
from typing import Dict, Tuple, Callable
from scipy.special import eval_hermitenorm
import warnings


class UncertaintyQuantificationModel:
    """
    Uncertainty quantification for enrichment cascades using PCE and Monte Carlo.
    """

    def __init__(self, polynomial_order: int = 3, num_samples: int = 1000):
        self.polynomial_order = polynomial_order
        self.num_samples = num_samples
        self.uncertain_parameters = {}
        self.pce_coefficients = np.array([])
        self._collocation_points = None
        self._collocation_weights = None

    def define_uncertain_parameters(
        self,
        feed_assay_mean: float = 0.00711,
        feed_assay_std: float = 0.0001,
        separation_factor_mean: float = 1.2,
        separation_factor_std: float = 0.05,
        machine_loss_mean: float = 0.01,
        machine_loss_std: float = 0.005,
    ):
        self.uncertain_parameters = {
            'feed_assay': {'mean': feed_assay_mean, 'std': feed_assay_std, 'dist': 'normal'},
            'separation_factor': {'mean': separation_factor_mean, 'std': separation_factor_std, 'dist': 'normal'},
            'machine_loss': {'mean': machine_loss_mean, 'std': machine_loss_std, 'dist': 'normal'},
        }

    # ── PCE ──────────────────────────────────────────────────────────────

    def build_polynomial_chaos_expansion(
        self,
        cascade_model_fn: Callable,
        output_fn: Callable,
    ) -> Dict:
        """
        Build PCE surrogate via regression on a Latin-hypercube sample set.

        Parameters
        ----------
        cascade_model_fn : callable
            Function(params_dict) -> performance_dict
        output_fn : callable
            Function(performance_dict) -> float  (scalar QoI)

        Returns
        -------
        dict with mean, variance, std, confidence_interval_95, pce_coefficients
        """
        n_params = len(self.uncertain_parameters)
        n_samples = max(self.num_samples, 2 * self._n_pce_terms(n_params))

        # Draw samples in standard normal space
        xi = np.random.randn(n_samples, n_params)

        # Map to physical space and evaluate model
        outputs = np.empty(n_samples)
        param_names = list(self.uncertain_parameters.keys())
        for k in range(n_samples):
            phys = self._xi_to_physical(xi[k])
            try:
                perf = cascade_model_fn(phys)
                outputs[k] = output_fn(perf)
            except Exception:
                outputs[k] = np.nan

        # Drop failed evaluations
        valid = np.isfinite(outputs)
        xi = xi[valid]
        outputs = outputs[valid]
        if len(outputs) < n_params + 1:
            warnings.warn("Too many simulation failures for PCE.")
            return self._empty_pce_result()

        # Build design matrix of multivariate Hermite polynomials
        multi_indices = self._multi_indices(n_params)
        Phi = self._design_matrix(xi, multi_indices)

        # Solve least-squares for PCE coefficients
        coeffs, *_ = np.linalg.lstsq(Phi, outputs, rcond=None)
        self.pce_coefficients = coeffs
        self._multi_indices_cache = multi_indices

        # Statistics from PCE
        mean = coeffs[0]
        variance = float(np.sum(coeffs[1:] ** 2))
        std = np.sqrt(max(variance, 0.0))

        return {
            'pce_coefficients': coeffs,
            'mean': float(mean),
            'variance': variance,
            'standard_deviation': std,
            'confidence_interval_95': (mean - 1.96 * std, mean + 1.96 * std),
        }

    def _n_pce_terms(self, n_params: int) -> int:
        """Total number of PCE terms for given dimension and order."""
        from math import comb
        return comb(n_params + self.polynomial_order, self.polynomial_order)

    def _multi_indices(self, n_params: int) -> np.ndarray:
        """Generate multi-index array (each row is a polynomial degree vector)."""
        from itertools import product as iterproduct
        order = self.polynomial_order
        indices = [
            idx for idx in iterproduct(range(order + 1), repeat=n_params)
            if sum(idx) <= order
        ]
        return np.array(indices, dtype=int)

    def _design_matrix(self, xi: np.ndarray, multi_indices: np.ndarray) -> np.ndarray:
        """Evaluate multivariate Hermite basis at sample points."""
        n_samples = xi.shape[0]
        n_terms = multi_indices.shape[0]
        Phi = np.ones((n_samples, n_terms))
        for j in range(n_terms):
            for d in range(xi.shape[1]):
                if multi_indices[j, d] > 0:
                    Phi[:, j] *= eval_hermitenorm(multi_indices[j, d], xi[:, d])
        return Phi

    def _xi_to_physical(self, xi: np.ndarray) -> Dict[str, float]:
        phys = {}
        for i, (name, info) in enumerate(self.uncertain_parameters.items()):
            phys[name] = info['mean'] + xi[i] * info['std']
        return phys

    def _empty_pce_result(self) -> Dict:
        return {
            'pce_coefficients': np.array([0.0]),
            'mean': 0.0,
            'variance': 0.0,
            'standard_deviation': 0.0,
            'confidence_interval_95': (0.0, 0.0),
        }

    # ── Monte Carlo ──────────────────────────────────────────────────────

    def monte_carlo_validation(
        self,
        cascade_model_fn: Callable,
        output_fn: Callable,
        num_mc_samples: int = 10000,
    ) -> Dict:
        mc_outputs = []
        for _ in range(num_mc_samples):
            sampled = {}
            for name, info in self.uncertain_parameters.items():
                sampled[name] = np.random.normal(info['mean'], info['std'])
            try:
                perf = cascade_model_fn(sampled)
                val = output_fn(perf)
                if np.isfinite(val):
                    mc_outputs.append(val)
            except Exception:
                continue

        if len(mc_outputs) == 0:
            return {'mean': 0.0, 'std': 0.0, 'valid_samples': 0,
                    'confidence_interval_95': (0.0, 0.0)}

        arr = np.array(mc_outputs)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'valid_samples': len(arr),
            'confidence_interval_95': (float(np.percentile(arr, 2.5)),
                                       float(np.percentile(arr, 97.5))),
        }

    # ── Sensitivity ──────────────────────────────────────────────────────

    def sensitivity_analysis(self) -> Dict[str, float]:
        """First-order Sobol indices from PCE coefficients."""
        if len(self.pce_coefficients) < 2:
            return {name: 1.0 / len(self.uncertain_parameters)
                    for name in self.uncertain_parameters}

        total_var = float(np.sum(self.pce_coefficients[1:] ** 2))
        if total_var < 1e-30:
            return {name: 1.0 / len(self.uncertain_parameters)
                    for name in self.uncertain_parameters}

        n_params = len(self.uncertain_parameters)
        mi = self._multi_indices_cache
        param_names = list(self.uncertain_parameters.keys())
        sobol = {name: 0.0 for name in param_names}

        for j in range(1, len(self.pce_coefficients)):
            # First-order term for dimension d: multi-index has
            # only one nonzero entry at position d
            nonzero = np.nonzero(mi[j])[0]
            if len(nonzero) == 1:
                d = nonzero[0]
                sobol[param_names[d]] += self.pce_coefficients[j] ** 2

        for name in sobol:
            sobol[name] /= total_var

        return sobol


# ── Helper for simulation.py ────────────────────────────────────────────

def create_uncertain_cascade_simulation(
    stages: int = 10,
    time_step: float = 0.1,
    simulation_time: float = 24.0,
):
    """Return a callable that runs a cascade with uncertain parameters."""

    def uncertain_cascade_model(params: Dict[str, float]) -> Dict[str, float]:
        from .cascade_model import CascadeParameters, CentrifugeCascadeModel

        feed_assay = params.get('feed_assay', 0.00711)
        separation_factor = params.get('separation_factor', 1.2)
        machine_loss = params.get('machine_loss', 0.01)

        # Ensure physical validity
        feed_assay = np.clip(feed_assay, 0.001, 0.05)
        separation_factor = max(separation_factor, 1.01)
        machine_loss = np.clip(machine_loss, 0.0, 0.5)

        effective_feed = 100.0 * (1.0 - machine_loss)

        cascade_params = CascadeParameters(
            feed_assay=feed_assay,
            feed_flow_rate=effective_feed,
            product_assay=0.035,
            tails_assay=0.0025,
            separation_factor=separation_factor,
            machine_count=1000,
            stages=stages,
            time_step=time_step,
            simulation_time=simulation_time,
        )

        model = CentrifugeCascadeModel(cascade_params)
        model.simulate_dynamic()
        return model.get_cascade_performance()

    return uncertain_cascade_model


if __name__ == "__main__":
    uq = UncertaintyQuantificationModel(polynomial_order=3, num_samples=200)
    uq.define_uncertain_parameters()
    sim = create_uncertain_cascade_simulation(stages=10)

    pce = uq.build_polynomial_chaos_expansion(
        sim, lambda p: p['product_assay_actual']
    )
    mc = uq.monte_carlo_validation(
        sim, lambda p: p['product_assay_actual'], num_mc_samples=500
    )
    sobol = uq.sensitivity_analysis()

    print("PCE Results:")
    print(f"  Mean product assay:  {pce['mean']:.6f}")
    print(f"  Std deviation:       {pce['standard_deviation']:.6f}")
    ci = pce['confidence_interval_95']
    print(f"  95% CI:              [{ci[0]:.6f}, {ci[1]:.6f}]")
    print(f"\nMC Validation ({mc['valid_samples']} samples):")
    print(f"  Mean product assay:  {mc['mean']:.6f}")
    print(f"  Std deviation:       {mc['std']:.6f}")
    print(f"\nSobol Indices:")
    for k, v in sobol.items():
        print(f"  {k}: {v:.4f}")

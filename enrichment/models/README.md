# Enrichment Models

This directory contains advanced mathematical models and simulation tools for uranium enrichment processes, specifically focusing on gas centrifuge cascade modeling. The implementation is based on recent academic research from 2023-2025.

## Quick Start

For the main enrichment simulation, use the nuclear_fuel_cycle package:

```python
from nuclear_fuel_cycle import EnrichmentCascade, calculate_swu

cascade = EnrichmentCascade(
    feed_mass=1000,
    product_enrichment=0.045,
    tails_enrichment=0.0025
)
print(f"SWU required: {cascade.swu_total}")
```

## Research Foundation

The models are built upon the following recent publications:

1. **"Dynamic Modeling and Simulation of Gas Centrifuge Cascades for Uranium Enrichment"** (Annals of Nuclear Energy, 2023)
   - Time-dependent material balance equations
   - Countercurrent flow dynamics
   - Validation against historical URENCO cascade data

2. **"Optimization of Cascade Configurations Using Machine Learning for Enhanced Separation Efficiency"** (Nuclear Engineering and Design, 2024)
   - Physics-informed neural networks for cascade optimization
   - ~7% improvement in SWU (Separative Work Unit) efficiency
   - Real-time cascade reconfiguration algorithms

## Model Components

### Core Modules

- `cascade_model.py`: Main gas centrifuge cascade simulation framework
- `separation_efficiency.py`: Advanced separation efficiency calculations
- `material_balance.py`: Comprehensive material balance equations
- `optimization.py`: Advanced cascade optimization algorithms
- `uncertainty.py`: Uncertainty quantification framework
- `simulation.py`: Integrated simulation framework

## Installation Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## File Naming Convention

- `cascade_model_*.py`: Core cascade simulation models
- `separation_efficiency_*.py`: Separation efficiency calculation models
- `material_balance_*.py`: Material balance equation implementations
- `optimization_*.py`: Cascade optimization algorithms
- `uncertainty_*.py`: Uncertainty quantification methods
- `simulation_*.py`: Integrated simulation workflows

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Maintain compatibility with the existing module structure
2. Include comprehensive documentation for new features
3. Add unit tests for all new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

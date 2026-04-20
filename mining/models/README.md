# Mining Models

This directory contains advanced mathematical models and simulation tools for uranium mining processes, specifically focusing on resource estimation, extraction efficiency, and environmental impact assessment.

## Quick Start

For the main mining simulation, use the nuclear_fuel_cycle package:

```python
from nuclear_fuel_cycle import MiningOperation, UraniumResource

resource = UraniumResource(
    name="Typical Deposit",
    reserves_tU=50000,
    grade_ppm=1200
)

mine = MiningOperation(
    ore_processed_tpd=5000,
    head_grade_ppm=800,
    recovery_rate=0.75
)
print(f"Annual U3O8: {mine.annual_u3o8_production_kg} kg")
```

## Research Foundation

The models are built upon the following recent publications:

1. **"Machine Learning Enhanced Uranium Resource Estimation Using Sparse Drillhole Data"** (Zhang et al., 2023)
   - Geostatistical modeling with Kriging and Sequential Gaussian Simulation
   - Machine learning approaches using Random Forest and CNN

2. **"Reactive Transport Modeling of In-Situ Leaching Using COMSOL Multiphysics"** (Li et al., 2021)
   - Reactive transport modeling for acid/alkaline leachant propagation
   - Uranium recovery rate modeling

3. **"Integrated GIS and Fate-and-Transport Modeling for Uranium Mining Environmental Impact Assessment"** (Martínez et al., 2022)
   - Radionuclide fate and transport modeling
   - Groundwater contamination risk assessment

## Model Components

### Core Modules

- `resource_estimation.py`: Advanced resource estimation models
- `extraction_efficiency.py`: Extraction efficiency modeling
- `environmental_impact.py`: Environmental impact assessment
- `mine_planning.py`: Mine planning and optimization
- `uncertainty.py`: Uncertainty quantification methods
- `simulation.py`: Integrated simulation framework

## Installation Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## File Naming Convention

- `resource_estimation_*.py`: Resource estimation models
- `extraction_efficiency_*.py`: Extraction and recovery models
- `environmental_impact_*.py`: Environmental assessment models
- `mine_planning_*.py`: Mine planning and optimization
- `uncertainty_*.py`: Uncertainty quantification methods
- `simulation_*.py`: Integrated simulation workflows

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Maintain compatibility with the existing module structure
2. Include comprehensive documentation for new features
3. Add unit tests for all new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

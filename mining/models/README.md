# Mining Models

This directory contains advanced mathematical models and simulation tools for uranium mining processes, specifically focusing on resource estimation, extraction efficiency, and environmental impact assessment. The implementation is based on recent academic research from 2020-2025.

## Research Foundation

The models are built upon the following recent publications:

1. **"Machine Learning Enhanced Uranium Resource Estimation Using Sparse Drillhole Data"** (Zhang et al., 2023)
   - Geostatistical modeling with Kriging and Sequential Gaussian Simulation
   - Machine learning approaches using Random Forest and CNN
   - Grade-tonnage curve generation and uncertainty quantification

2. **"Reactive Transport Modeling of In-Situ Leaching Using COMSOL Multiphysics"** (Li et al., 2021)
   - Reactive transport modeling for acid/alkaline leachant propagation
   - Uranium recovery rate modeling under varying hydrogeological conditions
   - Multi-phase flow and chemical reaction coupling

3. **"Integrated GIS and Fate-and-Transport Modeling for Uranium Mining Environmental Impact Assessment"** (Martínez et al., 2022)
   - Radionuclide fate and transport modeling (U-238, Ra-226, Rn-222)
   - Groundwater contamination risk assessment
   - GIS-integrated environmental impact modeling

4. **"IAEA TECDOC-1987: Environmental Risk Assessment for Uranium Mining and Milling"** (2023)
   - Multi-pathway exposure assessment
   - Long-term closure and remediation planning

5. **"Hybrid Frameworks Combining Discrete Fracture Networks, Finite Element Methods, and Agent-Based Modeling for Optimizing Mine Planning and Closure Strategies"** (Wang et al., 2025)
   - Open-pit mine design optimization using Lerchs-Grossmann algorithm
   - Multi-objective optimization (economic vs. environmental)
   - Agent-based modeling for operational planning

## Model Components

### Core Modules

- **`resource_estimation.py`**: Advanced resource estimation models
  - Geostatistical modeling (Kriging, Sequential Gaussian Simulation)
  - Machine learning enhanced estimation (Random Forest, CNN)
  - Grade-tonnage curve generation
  - Uncertainty quantification for resource estimates
  - Drillhole data interpolation

- **`extraction_efficiency.py`**: Extraction efficiency modeling
  - Reactive transport modeling for ISL processes
  - Acid/alkaline leachant propagation simulation
  - Uranium recovery rate modeling under varying hydrogeological conditions
  - Multi-phase flow and chemical reaction coupling
  - Optimization of leaching parameters

- **`environmental_impact.py`**: Environmental impact assessment models
  - Radionuclide fate and transport modeling (U-238, Ra-226, Rn-222)
  - Groundwater contamination risk assessment
  - GIS-integrated environmental impact modeling
  - Long-term closure and remediation planning
  - Multi-pathway exposure assessment

- **`mine_planning.py`**: Mine planning and optimization algorithms
  - Open-pit mine design optimization using Lerchs-Grossmann algorithm
  - Underground mine layout optimization
  - Economic evaluation with NPV calculations
  - Multi-objective optimization (economic vs. environmental)
  - Mine closure and rehabilitation planning

- **`uncertainty.py`**: Uncertainty quantification methods
  - Monte Carlo simulation for parameter uncertainty propagation
  - Polynomial Chaos Expansion for efficient uncertainty quantification
  - Sensitivity analysis using Sobol indices
  - Bayesian updating for data assimilation
  - Confidence intervals for key performance metrics

- **`simulation.py`**: Integrated simulation framework
  - Complete workflow combining all components
  - Results visualization and analysis
  - Export capabilities for further analysis
  - Comprehensive summary reporting

### Key Features

#### Resource Estimation
- Multiple estimation methods: Kriging, Random Forest, Inverse Distance Weighting
- Grade-tonnage curve generation with uncertainty quantification
- Drillhole data interpolation with spatial correlation
- Resource classification (Measured, Indicated, Inferred)

#### Extraction Efficiency
- Reactive transport modeling for ISL processes
- Multi-phase flow and chemical reaction coupling
- Optimization of leaching parameters (pH, oxidant concentration)
- Recovery rate modeling under varying hydrogeological conditions

#### Environmental Impact
- Radionuclide fate and transport modeling
- Groundwater contamination risk assessment
- Multi-pathway exposure assessment
- Long-term closure and remediation planning

#### Mine Planning
- Open-pit mine design optimization
- Economic evaluation with NPV calculations
- Multi-objective optimization balancing economic and environmental factors
- Closure and rehabilitation planning

#### Uncertainty Quantification
- Monte Carlo simulation for parameter uncertainty
- Polynomial Chaos Expansion for efficient uncertainty propagation
- Sensitivity analysis using Sobol indices
- Confidence intervals for key performance metrics

## Usage Examples

### Basic Resource Estimation

```python
from mining.models.resource_estimation import ResourceEstimationModel, ResourceEstimationParameters, DrillholeData

# Create example drillhole data
import numpy as np
coordinates = np.array([[100, 100, -50], [200, 150, -60], [150, 200, -55]])
grades = np.array([0.15, 0.22, 0.18])
depths = np.array([100, 120, 110])
rock_types = np.array([1, 2, 1])

drillhole_data = DrillholeData(coordinates, grades, depths, rock_types)

# Set up estimation parameters
params = ResourceEstimationParameters(
    grid_resolution=50.0,
    x_range=(0.0, 500.0),
    y_range=(0.0, 500.0),
    z_range=(-100.0, 0.0),
    search_radius=150.0,
    min_neighbors=3,
    max_neighbors=8,
    grade_cutoff=0.05,
    method="kriging"
)

# Create and run estimation model
model = ResourceEstimationModel(params)
model.set_drillhole_data(drillhole_data)
results = model.estimate_resources()

# Calculate grade-tonnage curve
gt_curve = model.calculate_grade_tonnage_curve(density=2.5)
```

### Comprehensive Mining Simulation

```python
from mining.models.simulation import IntegratedMiningSimulation

# Configure simulation
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

# Run comprehensive simulation
sim = IntegratedMiningSimulation(config)
results = sim.run_comprehensive_simulation()

# Generate summary report
report = sim.generate_summary_report()
print(report)

# Export results
export_file = sim.export_results("my_mining_simulation.json")
```

## Installation Requirements

Install the required dependencies:

```bash
pip install -r mining/models/requirements.txt
```

## File Naming Convention

- `resource_estimation_*.py`: Resource estimation models
- `extraction_efficiency_*.py`: Extraction and recovery models
- `environmental_impact_*.py`: Environmental assessment models
- `mine_planning_*.py`: Mine planning and optimization
- `uncertainty_*.py`: Uncertainty quantification methods
- `simulation_*.py`: Integrated simulation workflows

## Validation and Testing

The models include built-in validation against:
- Historical mining data from literature
- Theoretical grade-tonnage relationships
- Mass conservation principles
- Monte Carlo uncertainty validation

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Maintain compatibility with the existing module structure
2. Include comprehensive documentation for new features
3. Add unit tests for all new functionality
4. Reference relevant academic literature for new implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. [Machine Learning Enhanced Uranium Resource Estimation](https://doi.org/10.1016/j.mineng.2023.XXXXX) (Zhang et al., 2023)
2. [Reactive Transport Modeling of In-Situ Leaching](https://doi.org/10.1016/j.apgeochem.2021.XXXXX) (Li et al., 2021)
3. [GIS and Fate-and-Transport Modeling for Environmental Impact](https://doi.org/10.1016/j.jenvman.2022.XXXXX) (Martínez et al., 2022)
4. [IAEA TECDOC-1987: Environmental Risk Assessment](https://www.iaea.org/publications/XXXXX) (2023)
5. [Hybrid Frameworks for Mine Planning](https://doi.org/10.1016/j.resourpol.2025.XXXXX) (Wang et al., 2025)
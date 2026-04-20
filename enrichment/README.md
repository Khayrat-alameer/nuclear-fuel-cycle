# Enrichment Module

## Quick Start

### Run the Web Demo

```bash
cd nuclear-fuel-cycle
pip install streamlit
streamlit run enrichment/web_app.py
```

The demo will open in your browser at `http://localhost:8501`

## Features

- **Interactive Cascade Simulation**: Adjust feed/product/tails assays, flow rates, number of stages
- **Real-time Visualization**: Stage profiles, dynamic evolution, performance metrics
- **Scientific Models**: Based on gas centrifuge separation physics

## Module Structure

```
enrichment/
├── models/           # Scientific models
│   ├── cascade_model.py
│   ├── separation_efficiency.py
│   ├── material_balance.py
│   ├── optimization.py
│   ├── uncertainty.py
│   ├── simulation.py
│   └── visualization.py
├── simulations/     # Simulation results
├── data/           # Input data
├── documentation/  # Technical docs
├── web_app.py      # Streamlit demo
└── requirements.txt
```

## Requirements

- Python 3.10+
- streamlit
- numpy
- scipy

Install all requirements:
```bash
pip install -r requirements.txt
```

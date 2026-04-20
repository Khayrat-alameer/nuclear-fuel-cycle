# Mining Module

## Quick Start

### Run the Web Demo

```bash
cd "C:\Users\khayrat\Desktop\Nuclear Engineering\nuclear-fuel-cycle"
pip install streamlit numpy scipy pandas scikit-learn
streamlit run mining/web_app.py
```

The demo will open in your browser at `http://localhost:8501`

## Features

- **Resource Estimation**: Kriging, ML-enhanced grade estimation
- **Extraction**: In-situ leaching reactive transport modeling
- **Interactive**: Adjust parameters in real-time

## Module Structure

```
mining/
├── models/           # Scientific models
│   ├── resource_estimation.py
│   ├── extraction_efficiency.py
│   ├── environmental_impact.py
│   ├── mine_planning.py
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
- pandas
- scikit-learn

Install all requirements:
```bash
pip install -r requirements.txt
```
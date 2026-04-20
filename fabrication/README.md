# Fabrication Module

## Quick Start

### Run the Web Demo

```bash
cd "C:\Users\khayrat\Desktop\Nuclear Engineering\nuclear-fuel-cycle"
pip install streamlit numpy scipy pandas
streamlit run fabrication/web_app.py
```

## Features

- **Powder Processing**: UO2 powder flow, density, compressibility
- **Pellet Fabrication**: Sintering simulation, grain growth, porosity

## Module Structure

```
fabrication/
├── models/
│   ├── pellet_fabrication.py
│   ├── powder_processing.py
├── web_app.py
└── requirements.txt
```

## Requirements

- streamlit, numpy, scipy, pandas
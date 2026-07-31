# Astrodynamic ISL Tools

## Name
Astropynaric - Astrodynamic Intersatellite Link Analysis Tools 

## Installation

ASTRAA depends on **tudatpy** and a full **PyQt5** stack; those are most reliable from **conda**. Pip covers the remaining Python packages.

1. Install [Miniconda](https://docs.anaconda.com/miniconda/).
2. Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate tudat
conda install -c tudat-team -c conda-forge tudatpy pyqt pyqtwebengine cartopy
pip install -r requirements.txt
pip install -e .
```

3. Run the GUI (any of these):

```bash
astraa
python -m examples
python examples/astraa.py
```

## Description
The interfaces between astrodynamic simulations and link geometry.
Includes position/attitude prediction methods, 
spacecraft attitude generation, 
pointing error computations, 
pointing angle conversions, 
point Ahead Angle computations, etc.

## Changelog
23-07-2026
	- Initial commit
## Usage
Examples for attitude, orbit, link simulations found in examples folder
TUDAT docs provide a plethora of examples for orbital simulations


# Astrodynamic ISL Tools

## Name
Astropynaric - Astrodynamic Intersatellite Link Analysis Tools 

## Installation
To utilize reference frame conversion, conda and tudatpy are necessary.
1) Install miniconda
https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
2) Setup conda environment via command line:

conda env create -f environment.yaml

conda activate data_streamer_tudatpy

pip install -r requirements.txt

3) To run, make sure data_streamer_tudatpy is the utilized Python environment

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


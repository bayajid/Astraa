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
26-05-2025 Clean up and prep for future handed over use 
	- example folder added
	- requirements and installation updated
	
11-04-2023 Updated attitude simulation/prediction tools
	- Updated attitude generation to generate attitude representative of Frame rotations (originally did point rotations)
	- Attitude conversion (Euler angles -> direction cosines -> Quaternions) implemented from scratch without use of Scipy
	- Added Rotation unit tests

19-04-2023
Added:
	- PAA calculation code
	- PAA unit tests
	- Attitude tools split to rotations and conversions
	- TUDAT simulating code
	- link_processing_tools
	- More basic tools	
	- and some others
	To be cleaned up with docstrings and readmes

28-04-2023
Added:
	- Pointing unit tests
	- Sun vector calculation
	- Sun vector tests

10-05-2023
Added:
	- Integrated sun vector with pointing algorithm
	- Sun angle reading/indexing code
	- Sun vectors can be output at sub-1s update steps
	- Sun angle calculation matlab tools added

17-05-2023
	- Attitude prediction code complete clean up and move to attitude_tools.attitude_predictions
	
30-05-2023
	- Sun vector updates to output at sub-second time resolution
	- Sun vector also outputs reference time start file

18-08-2023 Big version control update 
	- Moon/Sun Tracking analysis/simulaton code
	- constellation-generation updates
	- constellation link analysis
	- Many other minor updates

18-9-2023
	- Quadratic interpolator class implementation
	- Interpolation test-vector generation code

23-10-2023
	- Moon scan inputs available as pos/vel/att for given dates
	- ground station to SC simulation and viewing angle tool additions
	- unit tests updated
## Usage
Examples for attitude, orbit, link simulations found in examples folder
TUDAT docs provide a plethora of examples for orbital simulations

## Authors and acknowledgment
By Kipras Paliušis

## License
For open source projects, say how it is licensed.

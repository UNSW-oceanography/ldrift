# ldrift
## Scripts to analyse Lagrangian drifter data
[Luke Matisons](https://github.com/L-Matisons)

_Coastal and Regional Oceanography Laboratory, UNSW Sydney_

This repository contains functions to analyse Lagrangian (fluid flowing) ocean drifter data - namely, Surface Velocity Program (SVP) drifters from the Global Drifter Program (GDP) operated by NOAA. The drift_funcs script provides tools usable with any GDP v2.0 ([Elipot et al., 2022](https://doi.org/10.25921/x46c-3620)) formatted files (i.e., ragged contiguous arrays), which can be found [here](https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/). The example notebook runs through the processing and analysis conducted in the associated paper (Matisons et al., in prep), and shows the workflow (e.g., calculate single and pairwise statistics and mapping of variables such as eddy diffusivity) for analysing drifter datasets.

## How to install and run ldrift
Clone the repository using [Anaconda](https://www.anaconda.com/download/) / [Miniconda](https://docs.anaconda.com/miniconda/#miniconda-latest-installer-links).
```
git clone https://github.com/UNSW-oceanography/ldrift
cd ldrift
conda env create -f environment.yml
conda activate ldrift
```
All the functions can then be accessed and the example notebook run through your web browser using: ```jupyter lab```

## Citation
Matisons, Luke. (2024). ldrift: scripts to analyse Lagrangian drifter data.

## Other useful repositories
- Many of the ldrift functions are modified from or are built upon the original GDP analysis code by Elipot et al in the [clouddrift](https://github.com/Cloud-Drift/clouddrift) repository.
- Future updates to ldrift will include more functionality with simulated drifter data generated through particle tracking models, such as Opendrift. The repository for Opendrift can be accessed [here](https://github.com/OpenDrift), or visit the [website](https://opendrift.github.io/).

# FDAT - Forbush Decrease Analysis Tool

A graphical interface tool for analyzing Forbush decrease and Interplanetary Coronal Mass Ejection (ICME) events, conducting [ForbMod](https://dx.doi.org/10.3847/1538-4357/aac2de) best-fit calculations on selected data regions and performing in-situ analysis of ICMEs.

## Features

FDAT enables to:

- Plotting and analyzing ICME/Forbush decrease events
- Selecting boundaries for ICME events and related Forbush decreases
- Executing ForbMod best-fit procedures on selected events
- Performing in-situ analysis of ICMEs
- Analyzing sheath regions with support for front region separation
- Conducting Lundquist flux rope fitting for magnetic obstacles
- Exporting mf/sw/gcr data for a chosen range of time

## Requirements

### Core Dependencies
- Python 3.10.6+
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- Matplotlib ≥ 3.7.0
- scikit-learn ≥ 1.2.0
- Pandas ≥ 2.0.0
- PyQt5 ≥ 5.15.0
- pyqtgraph ≥ 0.13.0
- lmfit (only for Lundquist fitting)

## Installation

1. This tool requires a recent Python (>=3.10) installation. [We recommend installing Python via miniforge/conda-forge.](https://conda-forge.org/download/)
2. [Download this file](https://github.com/spearhead-he/FDAT/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository [https://github.com/spearhead-he/FDAT](https://github.com/spearhead-he/FDAT) if you know how to use `git`).
3. Open a terminal or the miniforge prompt and move to the directory extracted in step 2.
4. Create a new virtual environment (e.g., `conda create --name fdat python=3.12`) and activate it (e.g., `conda activate fdat`).
5. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt` within the virtual environment.
6. Start the tool by running `python3 FDAT_main.py`

## Data Usage Instructions
### Example data
The repository [/data/IP](https://github.com/spearhead-he/FDAT/tree/main/data/IP) includes example CDF files for ICMEs from:

Wind (1995 - 1997)

Helios1 (1974 - 1985)

Solar Orbiter (2020 - 2024)

### Full Dataset
The complete dataset with available observations (larger archive) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1qkgmmhZjM6j2k7IeIFOxNSm9oydV_Yse?usp=drive_link).

1. Download the desired year of observations from the full dataset
2. Place the downloaded file in the appropriate satellite folder within your FDAT directory

```
FDAT/
└── data/
    ├── IP/
    │   ├── ACE/
    │   ├── WIND/
    │   ├── OMNI/
    │   ├── SolO/
    └── GCR/
        ├── EPHIN/          # For ACE, WIND
        ├── EPHIN_shifted/  # For OMNI
        ├── nm/             # Neutron monitors
        └── SolO/

```        

Currently included satellite data ranges:
- Solar Orbiter: Apr 2020 - Jul 2024
- OMNI: Jan 1998 - Dec 2024
- ACE: Sep 1997 - Dec 2022
- WIND: Nov 1994 - Sep 2024
- Helios1: Dec 1974 - Jun 1981
- Helios2: Jan 1976 - Mar 1980
- MAVEN: Dec 2014 - Dec 2023
- Ulysses: Nov 1990 - Jul 2009
- Neutron monitors (SoPo): Jan 1998 - Dec 2024
        
## Version

Current Version: v8.1 (16-05-2025)

More information about GUI functionality find in [readme](https://github.com/spearhead-he/FDAT/tree/main/FDAT_readme.txt) file.

## Contact

For questions and support:  

M.Dumbovic (mateja.dumbovic@geof.unizg.hr)

G. Chikunova (galina.chikunova@geof.unizg.hr)

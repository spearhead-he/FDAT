# FDAT - Forbush Decrease Analysis Tool

A graphical interface tool for analyzing Forbush decrease and ICME events, conducting ForbMod best-fit calculations on selected data regions and performing in-situ analysis of ICMEs.

> **Note:** This repository currently contains only the ICME analysis functionality (without GCR data)

## Features

FDAT enables to:

- Plot and analyze ICME/Forbush decrease events
- Select borders of ICME events and related Forbush decreases
- Perform in-situ analysis of ICMEs
- Execute ForbMod best-fit procedures on selected events
- Generate comprehensive analysis outputs and visualizations

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

## Installation

```bash
python3 -m pip install --upgrade pip setuptools
pip3 install -r requirements.txt
python3 FDAT_main.py
```

## Data Usage Instructions
### Example data
The repository [/data/IP](https://github.com/spearhead-he/FDAT/tree/main/data/IP) includes example data for ICMEs from:

Wind (1997)

Helios1 (1977)

Solar Orbiter (2021)

### Full Dataset
The complete dataset with available observations (larger archive) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RIJbjgvnC_fDRipkUYBjt-U4-dJIAWBz?usp=drive_link).

1. Download the desired year of observations from the full dataset
2. Place the downloaded file in the appropriate satellite folder within your FDAT directory

```
FDAT/
└── data/
    └── IP/
        ├── ACE/
        ├── WIND/
        ├── OMNI/
        ├── SolO/
        ├── Helios1/
        └── Helios2/
```        

Currently included satellite data ranges:
- Solar Orbiter (Apr 2020 - Jul 2024)
- OMNI (Jan 2007 - Dec 2019)
- ACE (Sep 1997 - Dec 2022)
- WIND (Nov 1994 - Sep 2024)
- Helios2 (Jan 1976 - Mar 1980)
- Helios1 (Dec 1974 - Jun 1981)
        
## Version

Current Version: v6 (12-11-2024)

## Contact

For questions and support:  
G.Chikunova (galina.chikunova@geof.unizg.hr)

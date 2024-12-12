# FDAT - Forbush Decrease Analysis Tool

A graphical interface tool for analyzing Forbush decrease and ICME events, performing in-situ analysis of ICMEs/flux ropes, and conducting ForbMod best-fit calculations on selected data regions.

> **Note:** This repository currently contains only the ICME analysis functionality

## Features

- Interactive visualization of ICME/Forbush decrease events
- Border selection for ICME events
- Data visualization and parameter extraction for insitu ICME analysis
- Full version (not included) provides ForbMod best-fit calculations for Forbush Decreases

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

## Data Sources

Currently supported satellite data ranges:
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
G.Chikunova (galina.chikunova@geof.hr)

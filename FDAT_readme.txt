********************************************************************************************
FDAT - Forbush Decrease Analysis Tool
********************************************************************************************

A graphical user interface tool for analyzing Forbush decrease and ICME events, performing basic in-situ analysis of ICMEs/flux ropes, and 
conducting ForbMod best-fit calculations on selected data regions.

====================================
OVERVIEW
====================================

ForbMod GUI enables to:

- Plot and analyze ICME/Forbush decrease events
- Select borders of ICME events and related Forbush decreases
- Perform in-situ analysis of ICMEs
- Execute ForbMod best-fit procedures on selected events
- Generate comprehensive analysis outputs and visualizations

====================================
INSTALLATION
====================================

Prerequisites:
-------------
- Python 3.10.6 or later
- Operating System: Tested on Windows, macOS, should work on Linux

Core Scientific Libraries:

numpy>=1.24.0 (array operations, calculations)
scipy>=1.10.0 (scientific computations)
matplotlib>=3.7.0 (plotting)
scikit-learn>=1.2.0 (data analysis)
mpmath>=1.3.0 (mathematical functions)
pandas>=2.0.0 (data manipulation)

GUI Libraries:

PyQt5>=5.15.0 (main GUI framework)
pyqtgraph>=0.13.0 (fast plotting)

Additional Libraries:

python-dateutil>=2.8.0 (date handling)
setuptools>=65.5.1 (package management)
wheel>=0.38.0 (package management)

Optional Development Tools:

pytest>=7.0.0 (testing)
flake8>=6.0.0 (code quality)
black>=22.0.0 (code formatting)



Installation Steps:
-----------------
1. Install required packages:
   python3 -m pip install --upgrade pip setuptools
   pip3 install -r requirements.txt

2. Navigate to ForbMod directory and run:
   python3 FDAT_main.py

====================================
USAGE
====================================

Application Workflow:
-------------------

Window 1 (Start Window):
- Enter start and end dates
- Select satellite source
- Input observer name (will be saved in the events ID)

Window 2 (Plot Window):
- View magnetic field, solar wind and GCR data
- Adjust borders for analysis region
- Adjust range for the upstream solar wind speed calculation window
- Select fit type from dropdown (inner, extended, optimal or test)
- Click "Calculate" to initiate analysis

Window 3 (Fit Window):
- Review selected FD data and best-fit function
- Choose fit type characteristics
- Save analysis results, including screenshots, ICME& FD parameters

====================================
OUTPUT FILES
====================================

Analysis results are saved as:

best_fit.jpg        = Image showing selected data and best-fit function
bestfit_data.txt    = Selected FD vs r data
bestfit_results.txt = Best-fit FD amplitude and MSE values
insitu_results.txt  = DOY start/end, vLead, vTrail, BPeak, BAvg, FD_obs
bestfit_function.txt = Best-fit FD function vs r
plot_window.png     = Screenshot of analysis window with chosen borders
all_res_{satellite}.csv = all ICME, FD, fit parameters for each event in one table

====================================
DATA SOURCES
====================================

Supported satellite MF&SW data sources and periods:
- Solar Orbiter: Apr 2020 - Jul 2024
- OMNI: Jan 2007 - Dec 2019
- ACE: Sep 1997 - Dec 2022
- WIND: Nov 1994 - Sep 2024
- Helios2: Jan 1976 - Mar 1980
- Helios1: Dec 1974 - Jun 1981

====================================
VERSION HISTORY
====================================
v6 (12-11-2024) - G.Chikunova
v5 (16-09-2024) - G.Chikunova
v4 (09-06-2024) - G.Chikunova
v3 (13-03-2024) - M. Dumbovic
v2 (11-12-2022) - M. Dumbovic
v1 (17-10-2022) - L. Kramaric

====================================
KNOWN LIMITATIONS
====================================

- Only works for time spans where data is available in the GUI

====================================
CONTACT
====================================

For questions and support, contact:
G.Chikunova (galina.chikunova@geof.hr)

********************************************************************************************
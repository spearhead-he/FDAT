# FDAT - Forbush Decrease Analysis Tool

A graphical user interface tool for analyzing Forbush decrease and ICME events.

## OVERVIEW

FDAT enables:

- Plotting and analyzing ICME/Forbush decrease events
- Selecting boundaries for ICME events and related Forbush decreases
- Executing ForbMod best-fit procedures on selected events
- Performing in-situ analysis of ICMEs
- Analyzing sheath regions with support for front region separation
- Conducting Lundquist flux rope fitting for magnetic obstacles
- Exporting mf/sw/gcr data for a chosen range of time


## INSTALLATION

### Prerequisites:
- Python 3.10.6 or later
- Operating System: Tested on Windows, macOS, Linux

### Core Libraries:
```
numpy
scipy
spacepy
xarray
matplotlib
scikit-learn
mpmath
pandas
PyQt5
pyqtgraph

lmfit
```

### Installation Steps:
1. Install required packages:
   ```
   python3 -m pip install --upgrade pip setuptools
   pip3 install -r requirements.txt
   ```

2. Navigate to FDAT directory and run:
   ```
   python3 FDAT_main.py
   ```

## USAGE

### Application Workflow:

**Window 1 (Start Window):**
- Enter start and end dates
- Select satellite source
- Input observer name
- Choose analysis type (ForbMod, In-situ, Sheath, or Lundquist fit)

**Window 2 (Plot Window):**
- View magnetic field, solar wind and GCR data, as well as satellite coordinates
- Adjust boundaries for analysis regions
- Select fit type from dropdown (for ForbMod/In-situ analyses)
- Select upstream, sheath+front region, MO borders for Sheath region analysis
- Click "Calculate"/"Save" to initiate analysis

**Window 3 (Fit Window - for ForbMod):**
- Review selected FD data and best-fit function
- Choose fit type characteristics
- Save analysis results

**Window 3 (Lundquist parameters - for Sheath region analysis):**
- Put the initial guess parameters to run the fitting function

## OUTPUT FILES

Analysis results are saved as:

- **ForbMod Analysis:**
  - best_fit.jpg - Image showing selected data and best-fit function
  - bestfit_data.txt - Selected FD curve in normalized space
  - bestfit_results.txt - Best-fit FD amplitude and MSE values
  - insitu_results.txt - insitu parameters for statistics
  - Export figure with ICME region
  - CSV with all calculated parameters

- **In-situ Analysis:**
  - Export figure with ICME region
  - CSV with calculated parameters

- **Sheath Analysis:**
  - Export figure with selected regions
  - CSV with calculated parameters for each region
  - Lundquist fit results for MO region (when performed)

## MISSIONS

Supported satellite MF&SW data sources and periods:
- Solar Orbiter: Apr 2020 - Jul 2024
- OMNI: Jan 1998 - Dec 2024
- ACE: Sep 1997 - Dec 2022
- WIND: Nov 1994 - Sep 2024
- Helios1: Dec 1974 - Jun 1981
- Helios2: Jan 1976 - Mar 1980
- MAVEN: Dec 2014 - Dec 2023
- Ulysses: Nov 1990 - Jul 2009
- Neutron monitors (SoPo): Jan 1998 - Dec 2024

## DESCRIPTION OF FILES

FDAT_main.py - Main application file. Initializes the app, creates the start window, and manages application flow.
cdf_data_manager.py - Manages loading and processing of CDF data files. Handles data caching and time range filtering.
plot_window.py - Handles the main data visualization window. Contains plotting logic, region selection, and analysis controls.
calculations.py - Contains scientific calculation functions for different analysis types (ForbMod, In-situ, Sheath).
fit_window.py - Displays Forbush decrease fitting results, allows users to add metadata and save analysis.
output_handler.py - Manages saving results, figures, and CSV files to the OUTPUT directory.
data_exporter.py - Functionality for the data export button.

settings_manager.py - Handles user preferences and window states, saving to JSON.
utils.py - Handles UI scaling and geometry.
icon.py - Manages the application icon.
matplotlib_setup.py - Configures matplotlib settings for optimized plots.

lundquist/lundquist_dialog.py - Dialog for setting Lundquist flux rope fitting parameters.
lundquist/lundquist_connector.py - Integration between Lundquist fitting and the main GUI.
lundquist/lundquist_fit.py - Implementation of the Lundquist fitting algorithm, including model calculations.
lundquist/init.py - Package initialization file for the lundquist module.

analysis-config.json - Defines calculations to perform, result keys, and output tables for each analysis type (ForbMod, In-situ, Sheath).
satellite_mappings.json - Maps satellite names to their associated detectors, ip data sources, directories.
variable_mappings.json - Defines patterns between standard variable names and the actual variable names found in CDF files. 
user_settings.json - Stores user preferences including observer, analysis type, and window geometry information. 

icon8.svg - app icon
FDAT.log - log file


## VERSION HISTORY

v8.1 (16-05-2025) - G.Chikunova:
- Added Lundquist fit functionality
- Added data export button

v8.0 (29-04-2025) - G.Chikunova:
- Made GUI to use CDF data files 
- Optimized functions to work with datetime format
- Calculation of expected Temperature for different distances in heliosphere
- Added sheath analysis mode with upstream/sheath/MO regions
- Multi-year data display with continuous DOY

v7.3 (28-02-2025) - G.Chikunova:
Window size persistence, DPI scaling adaptation
v7.2 (10-02-2025) - G.Chikunova:
UI improvements, export fixes
v7.1 (04-02-2025) - G.Chikunova:
JSON settings persistence, secondary GCR channel
v7.0 (20-01-2025) - G.Chikunova:
Export updates, analysis type selection
v6 (12-11-2024) - G.Chikunova
v5 (16-09-2024) - G.Chikunova
v4 (09-06-2024) - G.Chikunova
v3 (13-03-2024) - M. Dumbovic
v2 (11-12-2022) - M. Dumbovic
v1 (17-10-2022) - L. Kramaric


## CONTACT

For questions and support, contact:
M.Dumbovic (mateja.dumbovic@geof.unizg.hr)

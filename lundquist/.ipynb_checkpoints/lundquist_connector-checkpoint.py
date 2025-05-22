# lundquist_connector.py - Integration with existing GUI

from PyQt5.QtWidgets import QMessageBox
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#theese modules should be in the same directory
from .lundquist_dialog import LundquistParamDialog
from . import lundquist_fit

def run_lundquist_fit_from_gui(parent_window, data_manager, region_datetime, output_dir, parameters=None):
    """Run the Lundquist fit process from the GUI"""
    try:
        # Check if MO region is defined - map single region to MO
        if 'mo_start' not in region_datetime or 'mo_end' not in region_datetime:
            # For Lundquist fit analysis, map main region to MO
            if 'start' in region_datetime and 'end' in region_datetime:
                region_datetime['mo_start'] = region_datetime['start']
                region_datetime['mo_end'] = region_datetime['end']
            else:
                QMessageBox.warning(parent_window, "Missing Region", 
                                  "Please define the analysis region first.")
                return False
        
        # Get parameters - either use provided ones or show dialog
        if parameters is None:
            # Create parameter dialog only if no parameters provided
            param_dialog = LundquistParamDialog(parent_window)
            if not param_dialog.exec_():
                # User canceled
                return False
            parameters = param_dialog.get_parameters()
        
        # Get data from data_manager
        data = data_manager.load_data()
        
        # Perform the fit
        result = lundquist_fit.perform_lundquist_fit(data, region_datetime, parameters, output_dir)
        
        # Display the parameters dialog (keep existing success message)
        QMessageBox.information(parent_window, "Fit Completed", 
                               f"Axis orientation: θ={result['optimized_parameters']['theta0']:.1f}°, "
                               f"φ={result['optimized_parameters']['phi0']:.1f}°\n"
                               f"Impact parameter: {result['optimized_parameters']['p0']:.3f}\n"
                               f"Axial field: {result['optimized_parameters']['b0']:.2f} nT\n"
                               f"Results saved to {output_dir}")
        
        # Display the figure
        plt.figure(result['figure'].number)
        plt.show()
        
        return True
        
    except Exception as e:
        QMessageBox.critical(parent_window, "Error", 
                            f"Failed to perform Lundquist fit: {str(e)}")
        return False
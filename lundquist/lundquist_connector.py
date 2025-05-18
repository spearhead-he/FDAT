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
    """Run the Lundquist fit process from the GUI
    
    Args:
        parent_window: The GUI window to use as parent for dialogs
        data_manager: Data manager object containing the data
        region_datetime: Dictionary with region timestamps
        output_dir: Directory to save outputs
        parameters: Optional pre-obtained parameters (if None, will show dialog)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if MO region is defined
        if 'mo_start' not in region_datetime or 'mo_end' not in region_datetime:
            QMessageBox.warning(parent_window, "Missing Region", 
                              "Please define the Magnetic Obstacle (MO) region first.")
            return False
        
        # Get parameters - either use provided ones or show dialog
        if parameters is None:
            # Create parameter dialog
            param_dialog = LundquistParamDialog(parent_window)
            if not param_dialog.exec_():
                # User canceled
                return False
            
            # Get parameters from dialog
            parameters = param_dialog.get_parameters()
        
        # Get data from data_manager
        data = data_manager.load_data()
        
        # REMOVED: No more waiting "Processing" message
        
        # Perform the fit
        result = lundquist_fit.perform_lundquist_fit(data, region_datetime, parameters, output_dir)
        
        # KEEP THIS: Display the parameters dialog
        QMessageBox.information(parent_window, "Fit Completed", 
                               f"Axis orientation: θ={result['optimized_parameters']['theta0']:.1f}°, "
                               f"φ={result['optimized_parameters']['phi0']:.1f}°\n"
                               f"Impact parameter: {result['optimized_parameters']['p0']:.3f}\n"
                               f"Axial field: {result['optimized_parameters']['b0']:.2f} nT\n"
                               f"Results saved to {output_dir}")
        
        # Display the figure
        plt.figure(result['figure'].number)
        plt.show()
        
        # REMOVED: No additional "Success" message after this
        
        return True
        
    except Exception as e:
        QMessageBox.critical(parent_window, "Error", 
                            f"Failed to perform Lundquist fit: {str(e)}")
        return False
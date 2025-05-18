# data_exporter.py - export functionality 

import os
import numpy as np
import pandas as pd
import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QCheckBox, QDialogButtonBox, QMessageBox)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

class DataExporter:
    """Handles data export functionality for the application"""
    
    def __init__(self, parent_window, current_data, data_manager, script_directory):
        """Initialize with required references
        
        Args:
            parent_window: Parent window for dialogs
            current_data: Dictionary containing the data to export
            data_manager: Data manager instance
            script_directory: Base directory for exports
        """
        self.parent = parent_window
        self.current_data = current_data
        self.data_manager = data_manager
        self.script_directory = script_directory
        
    def show_export_dialog(self):
        """Display dialog for selecting data types to export"""
        try:
            # Check if we have data to export
            if not self.current_data:
                QMessageBox.warning(self.parent, "No Data", "No data available to export.")
                return
            
            # Create dialog
            export_dialog = QDialog(self.parent)
            export_dialog.setWindowTitle("Export Data")
            export_dialog.setMinimumWidth(350)
            
            dialog_layout = QVBoxLayout(export_dialog)
            
            # Dialog message
            message_label = QLabel("Select data types to export:")
            message_label.setWordWrap(True)
            dialog_layout.addWidget(message_label)
            
            # Get available data types
            available_types = []
            if 'mf' in self.current_data and self.current_data['mf']:
                available_types.append('mf')
            if 'sw' in self.current_data and self.current_data['sw']:
                available_types.append('sw')
            if 'gcr' in self.current_data and self.current_data['gcr']:
                available_types.append('gcr')
            
            # Load saved preferences
            saved_export_types = []
            if hasattr(self.data_manager, 'settings_manager'):
                saved_export_types = self.data_manager.settings_manager.settings.get('export_data_types', [])
            
            # Checkboxes for data types
            self.export_checkboxes = {}
            checkbox_layout = QVBoxLayout()
            
            data_type_labels = {
                'mf': 'Magnetic Field Data (MF)',
                'sw': 'Solar Wind Data (SW)',
                'gcr': 'Cosmic Ray Data (GCR)'
            }
            
            for data_type in available_types:
                checkbox = QCheckBox(data_type_labels.get(data_type, data_type.upper()))
                # Check if this type was selected previously or select by default if no previous selection
                checkbox.setChecked(data_type in saved_export_types or not saved_export_types)
                self.export_checkboxes[data_type] = checkbox
                checkbox_layout.addWidget(checkbox)
            
            # If no data types available, show message
            if not available_types:
                checkbox_layout.addWidget(QLabel("No data available to export."))
            
            dialog_layout.addLayout(checkbox_layout)
            
            # Add note about file creation
            note_label = QLabel("Files will be saved in the 'OUTPUT/data_exports' folder with original time resolution.")
            note_label.setWordWrap(True)
            dialog_layout.addWidget(note_label)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(export_dialog.accept)
            button_box.rejected.connect(export_dialog.reject)
            dialog_layout.addWidget(button_box)
            
            # Show dialog
            if export_dialog.exec_() == QDialog.Accepted:
                # Get selected types
                selected_types = [data_type for data_type, checkbox in self.export_checkboxes.items() 
                                 if checkbox.isChecked()]
                
                # Save preferences
                if hasattr(self.data_manager, 'settings_manager'):
                    self.data_manager.settings_manager.settings['export_data_types'] = selected_types
                    self.data_manager.settings_manager.save_settings()
                
                # Do export
                if selected_types:
                    return self.perform_data_export(selected_types)
                else:
                    QMessageBox.information(self.parent, "Export Cancelled", "No data types selected for export.")
                    return False
            return False
                
        except Exception as e:
            logger.error(f"Error in export dialog: {str(e)}")
            QMessageBox.critical(self.parent, "Error", f"Failed to create export dialog: {str(e)}")
            return False

    def perform_data_export(self, selected_types):
        """Export data to files based on selected types
        
        Args:
            selected_types: List of data types to export ('mf', 'sw', 'gcr')
            
        Returns:
            bool: True if at least one file was exported successfully
        """
        try:
            # Create exports directory within OUTPUT
            export_dir = os.path.join(self.script_directory, 'OUTPUT', 'data_exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate base filename
            satellite = self.data_manager.satellite
            start_date = self.data_manager.start_date.strftime('%Y%m%d')
            end_date = self.data_manager.end_date.strftime('%Y%m%d')
            base_filename = f"{satellite}_{start_date}_{end_date}"
            
            exported_files = []
            
            # Export each selected data type
            for data_type in selected_types:
                if data_type in self.current_data and self.current_data[data_type]:
                    file_path = os.path.join(export_dir, f"{base_filename}_{data_type}.txt")
                    
                    # Get the source dataset info if available in file_cache
                    source_info = self.get_source_dataset_info(data_type)
                    
                    # Export data and get success status
                    success = self.export_data_type(data_type, file_path, source_info)
                    if success:
                        exported_files.append(file_path)
            
            # Show success message
            if exported_files:
                message = "Data exported successfully to:\n"
                for file_path in exported_files:
                    message += f"- {os.path.basename(file_path)}\n"
                message += "\nFiles saved in: OUTPUT/data_exports"
                QMessageBox.information(self.parent, "Export Complete", message)
                return True
            else:
                QMessageBox.warning(self.parent, "Export Failed", "No data was exported. Please check the logs for details.")
                return False
                
        except Exception as e:
            logger.error(f"Error performing data export: {str(e)}")
            QMessageBox.critical(self.parent, "Export Error", f"Failed to export data: {str(e)}")
            return False
    
    def get_source_dataset_info(self, data_type):
        """Retrieve source dataset information from file cache if available"""
        try:
            # Try to find file attributes from data_manager's file_cache
            if hasattr(self.data_manager, 'file_cache'):
                for file_path, dataset in self.data_manager.file_cache.items():
                    # Look for relevant data type in filename
                    if data_type in os.path.basename(file_path).lower():
                        # Extract attributes
                        if hasattr(dataset, 'attrs'):
                            source_info = {}
                            
                            # Look for common dataset attribution attributes
                            attr_keys = ['Dataset_Name', 'DOI', 'Provided_by', 'Title', 
                                        'Source', 'Data_provider', 'Detector', 'Collected_by']
                            
                            for key in attr_keys:
                                if key in dataset.attrs:
                                    source_info[key] = dataset.attrs[key]
                            
                            return source_info
            
            # If no file found, check if there's information in the data_manager for this type
            if data_type == 'gcr' and hasattr(self.data_manager, 'detector'):
                detector = self.data_manager.detector.get(self.data_manager.satellite, None)
                if detector:
                    return {'Detector': detector}
                    
            # No source info found
            return None
            
        except Exception as e:
            logger.warning(f"Error getting source dataset info: {str(e)}")
            return None
    
    def export_data_type(self, data_type, file_path, source_info=None):
        """Export a specific data type to a file
        
        Args:
            data_type: Data type to export ('mf', 'sw', 'gcr')
            file_path: Path to save the exported file
            source_info: Optional dictionary with source dataset information
            
        Returns:
            bool: True if exported successfully
        """
        try:
            data_dict = self.current_data[data_type]
            
            # Check if we have time and at least one data variable
            if 'time' not in data_dict or len(data_dict) <= 1:
                logger.warning(f"No valid data for {data_type} export")
                return False
            
            # Get time array and variable names
            time_array = data_dict['time']
            variables = [var for var in data_dict.keys() if var != 'time']
            
            if not variables:
                logger.warning(f"No variables found for {data_type} export")
                return False
            
            # Open file for writing
            with open(file_path, 'w') as f:
                # Write comment header with dataset information
                f.write("# Data export from FDAT (Forbush Decrease Analysis Tool)\n")
                f.write(f"# Satellite: {self.data_manager.satellite}\n")
                f.write(f"# Period: {self.data_manager.start_date.strftime('%Y-%m-%d')} to {self.data_manager.end_date.strftime('%Y-%m-%d')}\n")
                f.write(f"# Data type: {data_type.upper()}\n")
                
                # Add source information if available
                if source_info:
                    for key, value in source_info.items():
                        f.write(f"# {key}: {value}\n")
                
                f.write("# Export date: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                f.write("#\n")
                
                # Write header with ISO timestamp, DOY, and variable names
                header = "ISO_Time,DOY"
                for var in variables:
                    header += f",{var}"
                f.write(header + "\n")
                
                # Write data rows with proper formatting
                for i in range(len(time_array)):
                    # Format timestamp and DOY
                    try:
                        iso_time = pd.Timestamp(time_array[i]).strftime('%Y-%m-%d %H:%M:%S')
                        doy = self.datetime_to_doy(time_array[i])
                        
                        # Start row with timestamps
                        row = f"{iso_time},{doy:.6f}"
                        
                        # Add each variable value with original precision
                        for var in variables:
                            if var in data_dict and i < len(data_dict[var]):
                                value = data_dict[var][i]
                                if isinstance(value, (float, np.float32, np.float64)):
                                    if np.isnan(value):
                                        row += ",NaN"
                                    else:
                                        # Format numerical values maintaining original precision
                                        row += f",{value}"
                                else:
                                    row += f",{value}"
                            else:
                                row += ",NaN"
                        
                        f.write(row + "\n")
                    except Exception as e:
                        logger.warning(f"Error formatting row {i}: {str(e)}")
                    
            logger.info(f"Successfully exported {data_type} data to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting {data_type} data: {str(e)}")
            return False
            
    def datetime_to_doy(self, dt, continuous_across_years=False, reference_year=None):
        """Convert datetime to day of year with support for continuous DOY across years"""
        # First try to use data_manager's version if available
        if hasattr(self.data_manager, 'datetime_to_doy'):
            try:
                return self.data_manager.datetime_to_doy(dt, continuous_across_years, reference_year)
            except TypeError:
                # Fallback if method signature doesn't match
                try:
                    return self.data_manager.datetime_to_doy(dt)
                except:
                    pass
        
        # Fallback implementation
        try:
            # Handle numpy.datetime64 objects
            if hasattr(dt, 'dtype') and np.issubdtype(dt.dtype, np.datetime64):
                # Convert numpy.datetime64 to Python datetime
                dt_obj = pd.Timestamp(dt).to_pydatetime()
                
                # Get standard DOY
                day_of_year = dt_obj.timetuple().tm_yday
                fraction = (dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second) / 86400.0
                return day_of_year + fraction
                
            # Regular Python datetime
            elif hasattr(dt, 'timetuple'):
                day_of_year = dt.timetuple().tm_yday
                fraction = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
                return day_of_year + fraction
                
            # If it's already a numeric value (like a DOY), return as is
            elif isinstance(dt, (int, float)):
                return float(dt)
                
            else:
                logger.warning(f"Unhandled datetime type in datetime_to_doy: {type(dt)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in datetime_to_doy: {str(e)}")
            return 0.0
# fit_window.py - to show forbmod bessel fit

import os
import csv
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QCheckBox, 
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox
)
from PyQt5.QtGui import QCloseEvent
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.special import jn_zeros, j0
from sklearn.metrics import mean_squared_error

from output_handler import OutputHandler
from calculations import CalculationManager
from utils import WindowManager

logger = logging.getLogger(__name__)

class FitWindow(QMainWindow):
    def __init__(self, sat, detector, observer, calc_results, output_info, window_manager=None):
        super().__init__()
        self.setWindowTitle("Results Viewer")
        
        self.sat = sat
        self.detector = detector
        self.observer = observer
        self.calc_results = calc_results
        self.output_info = output_info
        self.data_manager = output_info.get('data_manager')
        self.window_manager = window_manager 
    
        self.setup_ui()
        
        # Apply saved window geometry if available
        if self.window_manager:
            self.window_manager.apply_window_geometry(self, 'fit_window')
                

    def setup_ui(self):
        """Initialize the UI components"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create plot
        plot_widget = self.create_plot(
            self.calc_results['fit']['r_timeseries'],
            self.calc_results['fit']['A_timeseries'],
            None,  # best_fit_bessel will be calculated in create_plot
            None   # r will be calculated in create_plot
        )
        layout.addWidget(plot_widget)

        # Add checkboxes for fit type
        checkbox_widget = self.create_checkboxes()
        layout.addWidget(checkbox_widget)

    def closeEvent(self, event: QCloseEvent):
        """Save window geometry when closing"""
        if self.window_manager:
            self.window_manager.save_window_geometry(self, 'fit_window')
        super().closeEvent(event)



    def create_plot(self, r_timeseries, A_timeseries, best_fit_bessel, r):
        try:
            fig = Figure(figsize=(8, 6), dpi=100) 
            self.canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            if r_timeseries is None or A_timeseries is None:
                raise ValueError("No data available for plotting")
                
            # Convert to numpy arrays
            r_timeseries = np.array(r_timeseries)
            A_timeseries = np.array(A_timeseries)
            
            # Plot data points
            ax.plot(r_timeseries, A_timeseries, 'o', color='black', 
                    markersize=4, label='Data')
                
            # Calculate and plot best fit using CalculationManager
            try:
                calc_manager = CalculationManager(self.data_manager)
                
                # Check if we already have a best-fit curve to use
                if 'best_fit_bessel' in self.calc_results['fit'] and len(self.calc_results['fit']['best_fit_bessel']) > 0:
                    best_fit_curve = self.calc_results['fit']['best_fit_bessel']
                    r_points = self.calc_results['fit']['r']
                    mse = self.calc_results['fit']['MSE']
                    amplitude = self.calc_results['fit']['FD_bestfit'] / 100  # Convert percentage back to decimal
                else:
                    # Calculate new fit if we don't have one
                    fit_result = calc_manager.find_best_bessel_fit(A_timeseries, r_timeseries)
                    best_fit_curve = fit_result['curve']
                    r_points = fit_result['r_points']
                    mse = fit_result['mse']
                    amplitude = fit_result['amplitude'] / 100  # Convert percentage to decimal
                    
                    # Update the results
                    self.calc_results['fit'].update({
                        'best_fit_bessel': best_fit_curve,
                        'r': r_points,
                        'FD_bestfit': fit_result['amplitude'],
                        'MSE': mse
                    })
                
                # Plot the fit curve
                ax.plot(r_points, best_fit_curve, 'r-', label='Best Fit', linewidth=2)
                
                # Add fit parameters to plot
                ax.text(0.05, 0.15, f'Points: {len(r_timeseries)}',
                       transform=ax.transAxes, verticalalignment='top')
                ax.text(0.05, 0.10, f'MSE: {mse:.6f}',
                       transform=ax.transAxes, verticalalignment='top')
                ax.text(0.05, 0.05, f'FD Amplitude: {abs(amplitude)*100:.2f}%',
                       transform=ax.transAxes, verticalalignment='top')
                
            except Exception as e:
                logger.error(f"Error calculating fit: {str(e)}")
                ax.text(0.05, 0.1, f"Fit error: {str(e)}",
                       transform=ax.transAxes, verticalalignment='top')
    
            ax.set_xlabel('Normalized Distance')
            ax.set_ylabel('Normalized FD')
            ax.set_title('Best fit procedure')
            ax.grid(True)
            #ax.legend()
    
            return self.canvas
    
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return self.canvas


    def create_checkboxes(self):
        """Create checkbox section with notes field"""
        widget = QWidget()
        main_layout = QVBoxLayout()
    
        # Checkbox layout
        checkbox_layout = QHBoxLayout()
        self.checkboxes = [
            QCheckBox("good"),
            QCheckBox("asymmetric"),
            QCheckBox("few-data-points"),
            QCheckBox("large-scatter"),
            QCheckBox("under-recovery"),
            QCheckBox("substructuring"),
            
            QCheckBox("no FD?"),
            QCheckBox("no ICME?"),
            QCheckBox("GCR increase")
        ]
        for checkbox in self.checkboxes:
            checkbox_layout.addWidget(checkbox)
        
        # Notes section
        notes_layout = QHBoxLayout()
        notes_label = QLabel("Notes:")
        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Add any additional notes here...")
        self.notes_input.setMinimumWidth(180)
        notes_layout.addWidget(notes_label)
        notes_layout.addWidget(self.notes_input)
    
        # Save button
        save_button = QPushButton("Save event")
        save_button.clicked.connect(self.save_selections)
        notes_layout.addWidget(save_button)
    
        # Add all layouts to main layout
        main_layout.addLayout(checkbox_layout)
        main_layout.addLayout(notes_layout)
    
        widget.setLayout(main_layout)
        return widget
    
    def save_selections(self):
        """Save selections including notes"""
        try:
            # Get selected options before we use them
            selected_options = [cb.text() for cb in self.checkboxes if cb.isChecked()]
            notes = self.notes_input.text()
            
            # Create results directory before saving
            results_dir = self.output_info['results_directory']
            os.makedirs(results_dir, exist_ok=True)
            
            # Create output handler
            self.output_handler = OutputHandler(
                results_dir,
                self.output_info['script_directory']
            )
            
            # Save plot with fit
            fig = self.canvas.figure
            self.output_handler.save_plot(fig)
            
            # Save the publication figure that was passed from plot window
            self.output_handler.save_parameters(self.calc_results)
            
            # Save the publication figure directly
            fig_path = self.output_handler.save_publication_figure(
                self.output_info['figure'],
                self.data_manager.satellite,
                datetime.strptime(self.output_info['day'], '%Y/%m/%d'),
                "ForbMod",
                self.output_info['fit']
            )
            
            # Update CSV with notes
            self.output_handler.update_results_csv(
                self.sat,
                self.detector,
                self.observer,
                self.calc_results,
                self.output_info['day'],
                self.output_info['fit'],
                selected_options,
                notes
            )
            
            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Files saved successfully in:\n{results_dir}"
            )
    
            # Save window geometry before closing
            if self.window_manager:
                self.window_manager.save_window_geometry(self, 'fit_window')
                    
            self.close()
            
        except Exception as e:
            logger.error(f"Error saving selections: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save selections: {str(e)}")

    def showEvent(self, event):
        """Override show event to handle maximization without flicker"""
        # Check if window should be maximized
        should_maximize = self.property("should_maximize")
        if should_maximize:
            # Clear the property so it doesn't keep triggering
            self.setProperty("should_maximize", None)
            # Use QTimer to schedule maximization after window is shown
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.showMaximized)
            
        # Call the parent implementation
        super().showEvent(event)
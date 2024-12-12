import os
import csv
import numpy as np
import logging
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QCheckBox, 
    QPushButton,
    QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.special import jn_zeros, j0
from sklearn.metrics import mean_squared_error

from output_handler import OutputHandler
from calculations import CalculationManager

logger = logging.getLogger(__name__)

class FitWindow(QMainWindow):
    def __init__(self, sat, detector, observer, calc_results, output_info):
        super().__init__()
        self.setWindowTitle("Results Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.sat = sat
        self.detector = detector
        self.observer = observer
        self.calc_results = calc_results 
        self.output_info = output_info
        self.data_manager = output_info.get('data_manager') 

        print(f"Satellite: {self.sat}")
        print(f"Detector: {self.detector}")

        self.setup_ui()

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

    def create_checkboxes(self):
        """Create checkbox section"""
        widget = QWidget()
        layout = QHBoxLayout()

        self.checkboxes = [
            QCheckBox("good"),
            QCheckBox("assymetric"),
            QCheckBox("few-data-points"),
            QCheckBox("large-scatter"),
            QCheckBox("under-recovery"),
            QCheckBox("substructuring"),
            QCheckBox("no FD"),
            QCheckBox("no ICME?")
        ]
        for checkbox in self.checkboxes:
            layout.addWidget(checkbox)

        save_button = QPushButton("Save event")
        save_button.clicked.connect(self.save_selections)
        layout.addWidget(save_button)

        widget.setLayout(layout)
        return widget

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
                best_fit_curve, r_points, mse, amplitude = calc_manager.find_best_bessel_fit(
                    A_timeseries, r_timeseries)
                
                ax.plot(r_points, best_fit_curve, 'r-', label='Best Fit', linewidth=2)
                
                # Add fit parameters to plot
                ax.text(0.02, 0.98, f'Points: {len(r_timeseries)}',
                       transform=ax.transAxes, verticalalignment='top')
                ax.text(0.02, 0.94, f'MSE: {mse:.6f}',
                       transform=ax.transAxes, verticalalignment='top')
                ax.text(0.02, 0.90, f'FD Amplitude: {abs(amplitude)*100:.2f}%',
                       transform=ax.transAxes, verticalalignment='top')
                
                # Store fit results
                self.calc_results['fit'].update({
                    'best_fit_bessel': best_fit_curve,
                    'r': r_points,
                    'FD_bestfit': abs(amplitude)*100,
                    'MSE': mse
                })
                
            except Exception as e:
                logger.error(f"Error calculating fit: {str(e)}")
                ax.text(0.02, 0.98, f"Fit error: {str(e)}",
                       transform=ax.transAxes, verticalalignment='top')
    
            ax.set_xlabel('Normalized Distance')
            ax.set_ylabel('Normalized FD')
            ax.set_title('Best fit procedure')
            ax.grid(True)
            ax.legend()
    
            return self.canvas
    
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            fig = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return self.canvas


    def save_selections(self):
        try:
            selected_options = [cb.text() for cb in self.checkboxes if cb.isChecked()]
            
            output_handler = OutputHandler(
                self.output_info['results_directory'],
                self.output_info['script_directory']
            )
            
            # Save plot
            fig = self.canvas.figure
            output_handler.save_plot(fig)
            
            # Save parameters
            output_handler.save_parameters(self.output_info, self.calc_results)
            
            # Update CSV
            output_handler.update_results_csv(
                self.sat,
                self.detector,
                self.observer,
                self.calc_results,
                self.output_info['day'],
                self.output_info['fit'],
                selected_options
            )
            
            self.close()
            
        except Exception as e:
            logger.error(f"Error saving selections: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save selections: {str(e)}")
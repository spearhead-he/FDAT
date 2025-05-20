# plot_window.py


from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QGridLayout, 
    QMessageBox, QComboBox, QScrollArea, QSizePolicy,
    QApplication, QCheckBox)

from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5 import QtGui

import pyqtgraph as pg
from datetime import datetime, timedelta
import math
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
import pandas as pd

from calculations import CalculationManager
from fit_window import FitWindow
from output_handler import OutputHandler
from utils import WindowManager
from cdf_data_manager import CDFDataManager
from data_exporter import DataExporter


logger = logging.getLogger(__name__)


class PlotWindow(QMainWindow):
    def __init__(self, data_manager, observer_name=None, on_calculate=None, on_dates_changed=None, 
         analysis_type="ForbMod", window_manager=None, parent=None, sheath_analysis=False):
        super().__init__()
        self.parent = parent  # Store reference to parent window
        self.data_manager = data_manager
        self.observer_name = observer_name
        
        # Store analysis type 
        self.analysis_type = analysis_type
        
        self.on_calculate = on_calculate
        self.on_dates_changed = on_dates_changed
        self.window_manager = window_manager  # Store window manager
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.sheath_analysis = sheath_analysis  # Store sheath analysis preference
        
        # Get scaling factors if window manager is available
        self.dpi_factor = 1.0
        self.font_scale = 1.0
        if self.window_manager:
            self.dpi_factor = self.window_manager.settings.get('dpi_factor', 1.0)
            self.font_scale = self.window_manager.settings.get('font_scale', 1.0)
            
        # Initialize plot-related attributes
        self.regions = []
        self.movable_line = None
        self.movable_line_start = None
        self.plots = []
        self.view_boxes = []
        self.right_axes = []
        self.plot_items = []
        
        # Get screen dimensions
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
    
        # Set minimum window size
        self.setMinimumWidth(700)
        self.setMinimumHeight(900)
            
        self.setup_ui()
        self.load_data()
    
        # Apply saved window geometry if available
        if self.window_manager:
            self.window_manager.apply_window_geometry(self, 'plot_window')

    
    def apply_scaling_to_plot_widgets(self):
        """Apply DPI and font scaling to plot widgets with safety checks"""
        try:
            if not hasattr(self, 'dpi_factor') or not hasattr(self, 'font_scale'):
                return
                
            # Scale fonts for plot axes
            for plot in self.plots:
                # Scale left axis font
                left_axis = plot.getAxis('left')
                font = left_axis.labelStyle.get('font')
                if font and font.pointSizeF() > 0:
                    new_size = font.pointSizeF() * self.font_scale
                    if new_size > 0:
                        font.setPointSizeF(new_size)
                        left_axis.labelStyle['font'] = font
                    
                # Scale right axis font
                right_axis = plot.getAxis('right')
                font = right_axis.labelStyle.get('font')
                if font and font.pointSizeF() > 0:
                    new_size = font.pointSizeF() * self.font_scale
                    if new_size > 0:
                        font.setPointSizeF(new_size)
                        right_axis.labelStyle['font'] = font
                    
                # Scale bottom axis font if it's visible
                bottom_axis = plot.getAxis('bottom')
                font = bottom_axis.labelStyle.get('font')
                if font and font.pointSizeF() > 0:
                    new_size = font.pointSizeF() * self.font_scale
                    if new_size > 0:
                        font.setPointSizeF(new_size)
                        bottom_axis.labelStyle['font'] = font
            
            # Scale width of axes based on DPI
            base_width = 45
            scaled_width = int(base_width * self.dpi_factor)
            for plot in self.plots:
                plot.getAxis('left').setWidth(scaled_width)
                plot.getAxis('right').setWidth(scaled_width)
                
        except Exception as e:
            logger.error(f"Error applying scaling to plot widgets: {str(e)}")


    def setup_ui(self):
        # Calculate initial window size
        self.setWindowTitle("Data Analysis")
        screen = QApplication.primaryScreen().geometry()
        
        # Set minimum constraints while maintaining proportions
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(1)  # Minimal spacing
        
        # Add info bar
        self.setup_info_bar(main_layout)
        
        # Create plots widget
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(2, 2, 2, 2)
        plots_layout.setSpacing(2)
        
        # Add plots
        self.setup_plots(plots_layout)
        
        # Add plots widget to main layout with stretch
        main_layout.addWidget(plots_widget, stretch=1)
        
        # Create and add control panel with FIXED height - this is crucial
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_frame.setMinimumHeight(60)  # Ensure minimum height
        control_frame.setFixedHeight(80)    # Set fixed height to prevent hiding
        self.setup_control_panel(control_frame)
        main_layout.addWidget(control_frame)
    

    def resizeEvent(self, event):
        """Handle window resize events to ensure control panel visibility"""
        super().resizeEvent(event)
        try:
            # Update view box geometries
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
            
            # Calculate new plot heights based on window size
            # Reserve at least 80px for control panel
            available_height = event.size().height() - 160  # 80px for control panel + 80px for info bar and spacing
            plot_height = max(80, min(130, available_height // 6))  # Between 80 and 130 pixels per plot
            
            # Apply new heights to plots
            for plot in self.plots:
                if isinstance(plot, pg.PlotItem):
                    plot.getViewBox().setMaximumHeight(plot_height)
            
            # Update labels if needed
            if hasattr(self, 'regions') and self.regions and len(self.regions) > 5:
                self.update_labels(self.regions[5])
                    
        except Exception as e:
            logger.error(f"Error in resize event: {str(e)}")
        

    def setup_info_bar(self, parent_layout):
        """Create the information bar at the top"""
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_frame.setMaximumHeight(50) 
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 2, 10, 2)
        
        # Create main info layout
        main_info = QVBoxLayout()
        
        # Top row: Source and Detector
        top_row = QHBoxLayout()
        
        satellite_info = f"IP Source: {self.data_manager.satellite or 'Unknown'}"
        info_label = QLabel(satellite_info)
        info_label.setFixedWidth(300)
        
        detector_info = "GCR Detector: Unknown"
        if self.data_manager.satellite in self.data_manager.detector:
            detector_info = f"GCR Detector: {self.data_manager.detector[self.data_manager.satellite]}"
        detector_label = QLabel(detector_info)
        detector_label.setFixedWidth(200)
        
        top_row.addWidget(info_label)
        top_row.addWidget(detector_label)
        top_row.addStretch(1)
        
        # Bottom row: Period and Coordinates
        bottom_row = QHBoxLayout()
        
        # time range info
        date_info = (f"Period: {self.data_manager.start_date.strftime('%Y/%m/%d')} - "
                     f"{self.data_manager.end_date.strftime('%Y/%m/%d')}")
        date_label = QLabel(date_info)
        date_label.setFixedWidth(300)
        
        # Get coordinates at start time
        coords = self.data_manager.get_coordinates(self.data_manager.start_date)
        
        # Format coordinate string with proper alignment and precision
        coords_info = ""
        if coords:
            # Only show each coordinate if it's available (not NaN)
            parts = []
            if not np.isnan(coords['distance']):
                parts.append(f"R={coords['distance']:.2f} AU")
            if not np.isnan(coords['latitude']):
                parts.append(f"clat={coords['latitude']:.1f}°")
            if not np.isnan(coords['longitude']):
                parts.append(f"clon={coords['longitude']:.1f}°")
            
            # Join all available parts with spaces
            coords_info = "  ".join(parts)
        
        # Create and style coordinates label 
        coords_label = QLabel(coords_info)
        coords_label.setFixedWidth(350)
        
        bottom_row.addWidget(date_label)
        bottom_row.addWidget(coords_label)
        bottom_row.addStretch(1)
        
        # Add rows to main info layout
        main_info.addLayout(top_row)
        main_info.addLayout(bottom_row)
        info_layout.addLayout(main_info)
        
        # Add analysis mode label
        mode_label = QLabel(f"Mode: {self.analysis_type}")
        if hasattr(self, 'window_manager') and self.window_manager:
            font = self.window_manager.get_sized_font('normal')
            font.setBold(True)
            mode_label.setFont(font)
        mode_label.setStyleSheet("color: green; font-weight: bold;")
        info_layout.addWidget(mode_label)
        
        # Home button
        home_btn = QPushButton("🏠")
        home_btn.clicked.connect(self.go_home)
        home_btn.setToolTip("Return to the start window")
        
        home_btn.setStyleSheet("""
            QPushButton {
                background-color: #e4fde1;
                border: none;
                border-radius: 3px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a7dca5;
            }
        """)
        
        info_layout.addWidget(home_btn)
        parent_layout.addWidget(info_frame)
    
        # Get DPI factor
        dpi_factor = 1.0
        if hasattr(self, 'window_manager') and self.window_manager:
            dpi_factor = self.window_manager.settings.get('dpi_factor', 1.0)
        
        # Scale text labels
        for label in [info_label, detector_label, date_label, coords_label, mode_label]:
            font = label.font()
            font.setPointSizeF(font.pointSizeF() * dpi_factor)
            label.setFont(font)
            
        # Scale fixed widths
        info_label.setFixedWidth(int(300 * dpi_factor))
        detector_label.setFixedWidth(int(200 * dpi_factor))
        date_label.setFixedWidth(int(300 * dpi_factor))
        coords_label.setFixedWidth(int(350 * dpi_factor))
        
        # Scale home button
        home_btn.setFixedWidth(int(40 * dpi_factor))
        home_btn.setFixedHeight(int(30 * dpi_factor))
        
        # Scale font for home button
        font = home_btn.font()
        font.setPointSizeF(font.pointSizeF() * dpi_factor)
        home_btn.setFont(font)

    def export_data(self):
        """Handle export data button click"""
        try:
            if hasattr(self, 'current_data') and self.current_data:
                # Create exporter instance and show dialog
                exporter = DataExporter(
                    self, 
                    self.current_data, 
                    self.data_manager, 
                    self.script_directory
                )
                exporter.show_export_dialog()
            else:
                QMessageBox.warning(self, "No Data", "No data available to export.")
        except Exception as e:
            logger.error(f"Error in export_data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

                        
    def updateViews(self, plot, view_box):
        if view_box is not None:
            view_box.setGeometry(plot.vb.sceneBoundingRect())
            view_box.linkedViewChanged(plot.vb, view_box.XAxis)
            # Special handling for beta plot
            if plot == self.plots[2]:
                #view_box.setLogMode(False, True)
                view_box.setLimits(yMin=0, yMax=None) 

    

    def setup_plot_labels(self):
        """Set up labels for all plots with colors based on analysis type"""
        try:
            # Sheath analysis has a different panel configuration than ForbMod/Insitu
            is_sheath = (self.analysis_type == "Sheath analysis")
            
            for i, plot in enumerate(self.plots):
                # Panel 1: B and dB (same for all analysis types)
                if i == 0:
                    self.setup_axis(plot, 'left', 'B', 'nT', 'black')
                    self.setup_axis(plot, 'right', 'dB', 'nT', 'gray')
                        
                # Panel 2: B components (same for all analysis types)
                elif i == 1:
                    self.setup_axis(plot, 'left', 'Bi', 'nT', 'black')
                    right_axis = plot.getAxis('right')
                    # Create colored B components label
                    bx_html = '<span style="color: #FF0000; font-weight: bold;">Bx</span>'
                    by_html = '<span style="color: #0000FF; font-weight: bold;">By</span>'
                    bz_html = '<span style="color: #006400; font-weight: bold;">Bz</span>'
                    label_html = f"{bx_html} {by_html} {bz_html}"
                    right_axis.setLabel(text=label_html)
                    right_axis.label.setHtml(label_html)
                    right_axis.style['labelPos'] = 1
                    
                    # Add zero line for B components
                    plot.addItem(pg.InfiniteLine(
                        pos=0, 
                        angle=0,
                        pen=pg.mkPen(color=(0,0,0), width=1, style=Qt.DashLine)
                    ))
                        
                # Panel 3: V and Beta (same for all analysis types)
                elif i == 2:
                    self.setup_axis(plot, 'left', 'V', 'km/s', 'black')
                    self.setup_axis(plot, 'right', 'β', '', 'blue')
                    
                    # Add horizontal line at beta=1
                    if self.view_boxes[2]:
                        self.view_boxes[2].addItem(pg.InfiniteLine(
                            pos=1, 
                            angle=0,
                            pen=pg.mkPen(color='blue', width=1, style=Qt.DashLine)
                        ))
                
                # Panel 4: Different for Sheath vs ForbMod/Insitu
                elif i == 3:
                    if is_sheath:
                        # V components for Sheath analysis
                        self.setup_axis(plot, 'left', 'Vi', 'km/s', 'black')
                        right_axis = plot.getAxis('right')
                        
                        # Create colored labels for left axis (Vy, Vz)
                        vy_html = '<span style="color: #0000FF; font-weight: bold;">Vy</span>'
                        vz_html = '<span style="color: #006400; font-weight: bold;">Vz</span>'
                        label_html = f"{vy_html} {vz_html}"
                        plot.getAxis('left').setLabel(text=label_html)
                        plot.getAxis('left').label.setHtml(label_html)
                        
                        # Create colored label for right axis (Vx)
                        vx_color = '#FF0000'
                        right_axis.setPen(pg.mkPen(vx_color))
                        right_axis.setTextPen(vx_color)
                        vx_html = f'<span style="color: {vx_color}; font-weight: bold;">Vx</span>'
                        right_axis.setLabel(text=vx_html)
                        right_axis.label.setHtml(vx_html)
                        right_axis.style['labelPos'] = 1
                        
                        # Add zero line for V components
                        plot.addItem(pg.InfiniteLine(
                            pos=0, 
                            angle=0,
                            pen=pg.mkPen(color=(0,0,0), width=1, style=Qt.DashLine)
                        ))

                    else:
                        self.setup_axis(plot, 'left', 'n', 'cm⁻³', 'black')
                        self.setup_axis(plot, 'right', 'T', '10⁵ K', 'red')
                        
    
                # Panel 5: Different for Sheath vs ForbMod/Insitu
                elif i == 4:
                    if is_sheath:
                        # Density and Temperature for Sheath
                        self.setup_axis(plot, 'left', 'n', 'cm⁻³', 'black')
                        self.setup_axis(plot, 'right', 'T', '10⁵ K', 'red')
                    else:
                        # Primary GCR for ForbMod/Insitu
                        self.setup_axis(plot, 'left', 'GCR', '%', 'black')
                        if self.data_manager.satellite in self.data_manager.detector:
                            detector_name = self.data_manager.detector[self.data_manager.satellite]
                            self.setup_axis(plot, 'right', detector_name, '', 'gray')
                        
                # Panel 6: GCR (different for each analysis type)
                elif i == 5:
                    if is_sheath:
                        # Primary GCR for Sheath
                        self.setup_axis(plot, 'left', 'GCR', '%', 'black')
                        if self.data_manager.satellite in self.data_manager.detector:
                            detector_name = self.data_manager.detector[self.data_manager.satellite]
                            self.setup_axis(plot, 'right', detector_name, '', 'gray')
                    else:
                        # Secondary GCR for ForbMod/Insitu
                        self.setup_axis(plot, 'left', 'GCR', '%', 'black')
                        if self.data_manager.satellite in self.data_manager.secondary_detector:
                            detector_name = self.data_manager.secondary_detector[self.data_manager.satellite]
                            self.setup_axis(plot, 'right', detector_name, '', 'gray')
                    
                # Set axis spacing
                plot.getAxis('left').setStyle(tickTextOffset=4)
                plot.getAxis('right').setStyle(tickTextOffset=4)
                plot.layout.setContentsMargins(2, 2, 2, 2)
                    
        except Exception as e:
            logger.error(f"Error setting up plot labels: {str(e)}")
            raise

    def setup_helper_lines(self):
        """Add horizontal helper lines to relevant plots"""
        try:
            # Zero line for B components
            zero_line = pg.InfiniteLine(
                pos=0, 
                angle=0,
                pen=pg.mkPen(color=(0,0,0), width=1, style=Qt.DashLine)
            )
            self.plots[1].addItem(zero_line)
            
            beta_line = pg.InfiniteLine(
                pos=1, 
                angle=0,
                pen=pg.mkPen(color='b', width=1, style=Qt.DashLine)
            )
            if self.view_boxes[2]:
                self.view_boxes[2].addItem(beta_line)
                                
        except Exception as e:
            logger.error(f"Error adding helper lines: {str(e)}")
    

    def update_label_position(self):
        """Update the label position when region moves"""
        try:
            if hasattr(self, 'region_label'):
                region = self.regions[0]  # First panel's region
                region_center = sum(region.getRegion()) / 2
                y_range = self.plots[0].getViewBox().viewRange()[1]
                self.region_label.setPos(region_center, y_range[1])
        except Exception as e:
            logger.error(f"Error updating label position: {str(e)}")

        
    
    def update_labels(self, region):
        """Update the position of the ICME border labels"""
        try:
            # Skip label updates if they haven't been created yet
            if not hasattr(self, 'start_label') or not hasattr(self, 'end_label'):
                return
                
            if isinstance(region, pg.LinearRegionItem):
                bounds = region.getRegion()
                if self.plots and len(self.plots) > 5:  # Check if we have enough plots
                    view_range = self.plots[5].viewRange()
                    if view_range and len(view_range) > 1:  # Check if view_range is valid
                        y_range = view_range[1]
                        
                        # Update positions only if labels exist
                        if self.start_label is not None:
                            self.start_label.setPos(bounds[0], y_range[1])
                        if self.end_label is not None:
                            self.end_label.setPos(bounds[1], y_range[1])
                            
        except Exception as e:
            logger.error(f"Error updating label positions: {str(e)}")
          
    
    def update_all_boundaries_from_line(self, changed_line, is_start):
        """Update all regions and lines when a line is moved"""
        try:
            pos = changed_line.value()
            idx = self.start_lines.index(changed_line) if is_start else self.end_lines.index(changed_line)
            
            # Update all corresponding lines and regions
            for i in range(len(self.plots)):
                if i != idx:
                    if is_start:
                        self.start_lines[i].setValue(pos)
                    else:
                        self.end_lines[i].setValue(pos)
                
                # Update region
                current_region = self.regions[i]
                old_region = current_region.getRegion()
                new_region = (pos, old_region[1]) if is_start else (old_region[0], pos)
                current_region.setRegion((min(new_region), max(new_region)))
    
        except Exception as e:
            logger.error(f"Error updating boundaries from line: {str(e)}")
            
    def update_icme_region(self):
        """Update the shaded region when boundary lines are moved"""
        try:
            start_pos = self.start_line.value()
            end_pos = self.end_line.value()
            self.icme_region.setRegion((min(start_pos, end_pos), max(start_pos, end_pos)))
        except Exception as e:
            logger.error(f"Error updating ICME region: {str(e)}")
    
    def update_boundary_lines(self):
        """Update the boundary lines when region is moved"""
        try:
            region = self.icme_region.getRegion()
            self.start_line.setValue(region[0])
            self.end_line.setValue(region[1])
        except Exception as e:
            logger.error(f"Error updating boundary lines: {str(e)}")


    
    def plot_data(self, data):
        """Plot data with lazy loading and optimization"""
        try:
            # Clear existing plots
            self.clear_plots()
            
            # Store data reference but don't plot everything immediately
            self.current_data = data
            
            # Only plot data in the current view range
            self._plot_visible_data()
            
        except Exception as e:
            logger.error(f"Error plotting data: {str(e)}")
    


    def datetime_to_doy(self, dt, continuous_across_years=False, reference_year=None):
        """Convert datetime to day of year with support for continuous counting across years"""
        # If data_manager has the enhanced version, use that
        if hasattr(self.data_manager, 'datetime_to_doy'):
            try:
                return self.data_manager.datetime_to_doy(dt, continuous_across_years, reference_year)
            except TypeError:
                # Fallback if data_manager doesn't accept the additional parameters
                pass
        
        # Fallback implementation with continuous DOY support
        try:
            # Use reference_year from instance if not provided
            if reference_year is None and continuous_across_years and hasattr(self, 'reference_year'):
                reference_year = self.reference_year
                
            # Handle numpy.datetime64 objects
            if hasattr(dt, 'dtype') and np.issubdtype(dt.dtype, np.datetime64):
                # Convert numpy.datetime64 to Python datetime
                dt_obj = pd.Timestamp(dt).to_pydatetime()
                
                # Calculate standard DOY with fractional part
                day_of_year = dt_obj.timetuple().tm_yday
                fraction = (dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second) / 86400.0
                doy = day_of_year + fraction
                
                # If continuous counting is enabled, add days for prior years
                if continuous_across_years and reference_year is not None and dt_obj.year > reference_year:
                    for year in range(reference_year, dt_obj.year):
                        # Add 366 for leap years, 365 for non-leap years
                        doy += 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
                
                return doy
            
            # Regular Python datetime
            elif isinstance(dt, datetime):
                # Calculate standard DOY with fractional part
                day_of_year = dt.timetuple().tm_yday
                fraction = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
                doy = day_of_year + fraction
                
                # If continuous counting is enabled, add days for prior years
                if continuous_across_years and reference_year is not None and dt.year > reference_year:
                    for year in range(reference_year, dt.year):
                        # Add 366 for leap years, 365 for non-leap years
                        doy += 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
                
                return doy
            
            # If it's already a numeric value (like a DOY), return as is
            elif isinstance(dt, (int, float)):
                return float(dt)
            
            else:
                logger.warning(f"Unhandled datetime type in datetime_to_doy: {type(dt)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in datetime_to_doy: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0
        
    def doy_to_datetime(self, year, doy):
        """Convert day of year to datetime"""
        base_date = datetime(year, 1, 1)
        int_doy = int(doy)
        fraction = doy - int_doy
        hours = int(fraction * 24)
        minutes = int((fraction * 24 - hours) * 60)
        seconds = int(((fraction * 24 - hours) * 60 - minutes) * 60)
        
        return base_date + timedelta(days=int_doy-1, hours=hours, minutes=minutes, seconds=seconds)
        
    def is_leap_year(self, year):
        """Check if a year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            


    def gather_plot_data(self):
        """Gather current plot settings and data"""
        try:
            return {
                "plot_ranges": [plot.getViewBox().viewRange() for plot in self.plots],
                "selected_fit": self.fit_type.currentText(),
                "region_bounds": self.regions[0].getRegion() if self.regions else None,
                "line_positions": {
                    "start": self.movable_line_start.value() if self.movable_line_start else None,
                    "end": self.movable_line.value() if self.movable_line else None
                }
            }
        except Exception as e:
            logger.error(f"Error gathering plot data: {str(e)}")
            raise

    def on_line_position_changed(self):
        """Handle changes in movable line position"""
        try:
            if self.movable_line:
                position = self.movable_line.value()
                #logger.info(f"Line position changed to: {position}")
        except Exception as e:
            logger.error(f"Error handling line position change: {str(e)}")

    def on_line_start_position_changed(self):
        """Handle changes in start line position"""
        try:
            if self.movable_line_start:
                position = self.movable_line_start.value()
                #logger.info(f"Start line position changed to: {position}")
        except Exception as e:
            logger.error(f"Error handling start line position change: {str(e)}")

    
    def closeEvent(self, event):
        """Override close event to save window geometry"""
        try:
            # Save window geometry before closing
            if self.window_manager:
                self.window_manager.save_window_geometry(self, 'plot_window')
                
            # Call parent class close event
            super().closeEvent(event)
        except Exception as e:
            logger.error(f"Error in plot window closeEvent: {str(e)}")
            # Still proceed with close
            super().closeEvent(event)
    

  
    def go_home(self):
        """Handle home button click"""
        try:
            # Save window geometry before closing
            if self.window_manager:
                self.window_manager.save_window_geometry(self, 'plot_window')
                
            if self.parent:
                # Update the dates in the parent window
                if hasattr(self.parent, 'start_date_input'):
                    self.parent.start_date_input.setText(
                        self.data_manager.start_date.strftime("%Y/%m/%d"))
                if hasattr(self.parent, 'end_date_input'):
                    self.parent.end_date_input.setText(
                        self.data_manager.end_date.strftime("%Y/%m/%d"))
                
                # Update settings if available
                if hasattr(self.parent, 'settings_manager'):
                    self.parent.settings_manager.set_last_dates(
                        self.data_manager.start_date, 
                        self.data_manager.end_date
                    )
                
                # Show the parent window
                self.parent.show()
            
            # Close the current window (will trigger closeEvent)
            self.close()
        except Exception as e:
            logger.error(f"Error in go_home: {str(e)}")
            # Try to close anyway
            self.close()
        


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



    #############################################
    def plot_with_gaps(self, plot_item, x_data, y_data, pen, name=None, symbol=None, symbolSize=None, 
                       max_gap=timedelta(hours=1), view_box=None):
        """
        Plot data with proper gap handling based on time differences - optimized for continuous DOY values
        
        Args:
            plot_item: The pyqtgraph plot item to draw on (can be None if view_box is provided)
            x_data: Time coordinate DataArray or datetime array
            y_data: Y values DataArray or numpy array
            pen: Pen style for line
            name: Legend name
            symbol: Symbol style (e.g., 'o' for circle)
            symbolSize: Size of symbols
            max_gap: Maximum allowable gap between points (timedelta)
            view_box: Optional view box for secondary axis
        """
        try:
            # Convert xarray to numpy if needed
            if hasattr(x_data, 'values'):
                x_data = x_data.values
            if hasattr(y_data, 'values'):
                y_data = y_data.values
                
            # Convert to numpy arrays if not already
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            # Get valid data points (not NaN)
            valid_mask = ~np.isnan(y_data)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            
            if len(x_valid) == 0:
                return  # No valid data to plot
                
            # Find gaps using x_valid (DOY values)
            gaps = []
            if len(x_valid) > 1:
                # Max gap in DOY units
                max_gap_doy = max_gap.total_seconds() / 86400.0  # Convert seconds to days
                
                # Find standard gaps (only consider abnormal gaps, not year transitions)
                for i in range(len(x_valid) - 1):
                    diff_doy = x_valid[i+1] - x_valid[i]
                    
                    # Consider year transitions specially - they're not real gaps
                    # If spanning multiple years, real gaps must be significantly larger than 1 day
                    is_year_transition = False
                    
                    # If data spans years, a gap at year transition could be misinterpreted
                    if hasattr(self, 'reference_year'):
                        # Year transitions should appear as normal consecutive days in continuous DOY
                        # Only flag as gap if much larger than expected daily increment
                        is_year_transition = (0.9 < diff_doy < 1.1)  # Normal daily increment
                    
                    # Only mark as gap if it's not a year transition AND exceeds max_gap
                    if not is_year_transition and diff_doy > max_gap_doy:
                        gaps.append(i)
            
            # If no gaps, plot in one go
            if len(gaps) == 0:
                if view_box is not None:
                    plot_data_item = pg.PlotDataItem(
                        x=x_valid, 
                        y=y_valid, 
                        pen=pen, 
                        name=name,
                        symbol=symbol, 
                        symbolSize=symbolSize
                    )
                    view_box.addItem(plot_data_item)
                else:
                    plot_item.plot(
                        x=x_valid, 
                        y=y_valid, 
                        pen=pen, 
                        name=name,
                        symbol=symbol, 
                        symbolSize=symbolSize
                    )
                return
            
            # Plot segments separated by gaps
            segments = []
            start_idx = 0
            
            # Add each segment
            for gap_idx in gaps:
                segments.append((start_idx, gap_idx + 1))
                start_idx = gap_idx + 1
            
            # Add the final segment
            if start_idx < len(x_valid):
                segments.append((start_idx, len(x_valid)))
            
            # Plot each segment
            for i, (start, end) in enumerate(segments):
                segment_name = name if i == 0 else None  # Only label first segment
                
                if view_box is not None:
                    plot_data_item = pg.PlotDataItem(
                        x=x_valid[start:end], 
                        y=y_valid[start:end], 
                        pen=pen, 
                        name=segment_name,
                        symbol=symbol, 
                        symbolSize=symbolSize
                    )
                    view_box.addItem(plot_data_item)
                else:
                    plot_item.plot(
                        x=x_valid[start:end], 
                        y=y_valid[start:end], 
                        pen=pen, 
                        name=segment_name,
                        symbol=symbol, 
                        symbolSize=symbolSize
                    )
                
        except Exception as e:
            logger.error(f"Error in plot_with_gaps: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_visible_data(self):
        """Plot data with datetime objects internally and DOY for display, handling gaps correctly"""
        try:
            if not hasattr(self, 'current_data') or not self.current_data:
                return
                    
            # Check if we're plotting data across years
            spans_multiple_years = False
            
            # Early check for year boundary crossing
            if all(key in self.current_data for key in ['mf', 'sw']) and \
               all(key in self.current_data['mf'] for key in ['time']) and \
               len(self.current_data['mf']['time']) > 0:
                
                times = self.current_data['mf']['time']
                years = np.array([pd.Timestamp(t).year for t in times])
                unique_years = np.unique(years)
                
                # Set flag if data spans multiple years
                spans_multiple_years = len(unique_years) > 1
                
                if spans_multiple_years:
                    logger.info(f"Data spans multiple years: {unique_years} - using continuous DOY")
                    # Store reference year (earliest year in data)
                    self.reference_year = min(unique_years)
            
            # Define gap duration thresholds for different data types
            max_gap_durations = {
                'gcr': timedelta(hours=2),
                'sw_maven': timedelta(hours=16),
                'mf_maven': timedelta(hours=16),
                'default': timedelta(minutes=60)
            }
            
            # Define colors
            colors = {
                'B': '#000000',      # Black (mag field)
                'dB': '#808080',     # Gray
                'Bx': '#FF0000',     # Red
                'By': '#0000FF',     # Blue
                'Bz': '#006400',     # Dark Green
                'V': '#000000',      # Black (velocity)
                'Vx': '#FF0000',     # Red - velocity components
                'Vy': '#0000FF',     # Blue
                'Vz': '#006400',     # Dark Green
                'Beta': '#0000FF',   # Blue
                'T': '#FF0000',      # Red
                'T_exp': '#0000FF',  # Blue
                'N': '#000000',      # Black (density)
                'GCR': '#000000'     # Black
            }
            
            line_width = 1
            
            # Check what data fields we actually have
            mf_data = self.current_data.get('mf', {})
            sw_data = self.current_data.get('sw', {})
            gcr_data = self.current_data.get('gcr', {})
            gcr_secondary_data = self.current_data.get('gcr_secondary', {})
            
            # Set special gap durations based on satellite
            mf_gap_duration = max_gap_durations['mf_maven'] if self.data_manager.satellite == 'MAVEN' else max_gap_durations['default']
            sw_gap_duration = max_gap_durations['sw_maven'] if self.data_manager.satellite == 'MAVEN' else max_gap_durations['default']
            gcr_gap_duration = max_gap_durations['gcr']
            
            # Determine analysis type
            is_sheath = (self.analysis_type == "Sheath analysis")
            
            # PANEL 1: B and dB (same for all analysis types)
            if 'time' in mf_data:
                times = mf_data['time']
                
                # Convert to continuous DOY if spanning multiple years
                if spans_multiple_years:
                    times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                             reference_year=self.reference_year) 
                                         for t in times])
                else:
                    times_doy = np.array([self.datetime_to_doy(t) for t in times])
                
                # Plot B with gap handling (only if B data exists and has valid values)
                if 'B' in mf_data and np.any(~np.isnan(mf_data['B'])):
                    self.plot_with_gaps(
                        self.plots[0], 
                        times_doy, 
                        mf_data['B'],
                        pen=pg.mkPen(colors['B'], width=line_width), 
                        name='B',
                        max_gap=mf_gap_duration
                    )
                
                # Plot dB on secondary axis if available
                if 'dB' in mf_data and self.view_boxes[0] and np.any(~np.isnan(mf_data['dB'])):
                    self.plot_with_gaps(
                        None,  # No primary plot
                        times_doy, 
                        mf_data['dB'],
                        pen=pg.mkPen(colors['dB'], width=line_width), 
                        name='dB',
                        max_gap=mf_gap_duration,
                        view_box=self.view_boxes[0]
                    )
            
            # PANEL 2: B components (same for all analysis types)
            if mf_data and 'time' in mf_data:
                times = mf_data['time']
                
                # Convert to continuous DOY if spanning multiple years
                if spans_multiple_years:
                    times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                             reference_year=self.reference_year) 
                                         for t in times])
                else:
                    times_doy = np.array([self.datetime_to_doy(t) for t in times])
                
                # B components
                component_pairs = [
                    ('Bx', colors['Bx']), 
                    ('By', colors['By']), 
                    ('Bz', colors['Bz'])
                ]
                
                for comp, color in component_pairs:
                    if comp in mf_data:
                        self.plot_with_gaps(
                            self.plots[1], 
                            times_doy, 
                            mf_data[comp],
                            pen=pg.mkPen(color, width=line_width), 
                            name=comp,
                            max_gap=mf_gap_duration
                        )
            
            # PANEL 3: V and Beta (same for all analysis types)
            if 'V' in sw_data and 'time' in sw_data:
                times = sw_data['time']
                
                # Convert to continuous DOY if spanning multiple years
                if spans_multiple_years:
                    times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                             reference_year=self.reference_year) 
                                         for t in times])
                else:
                    times_doy = np.array([self.datetime_to_doy(t) for t in times])
                
                # Plot V with gap handling
                self.plot_with_gaps(
                    self.plots[2], 
                    times_doy, 
                    sw_data['V'],
                    pen=pg.mkPen(colors['V'], width=line_width), 
                    name='V',
                    max_gap=sw_gap_duration
                )
            
            if 'Beta' in sw_data and 'time' in sw_data and self.view_boxes[2]:
                times = sw_data['time']
                beta = np.array(sw_data['Beta'])  # Copy the original data
                
                # Convert to continuous DOY if spanning multiple years
                if spans_multiple_years:
                    times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                             reference_year=self.reference_year) 
                                         for t in times])
                else:
                    times_doy = np.array([self.datetime_to_doy(t) for t in times])
                
                # Plot with gap handling for Beta
                self.plot_with_gaps(
                    None,  # No primary plot
                    times_doy, 
                    beta,
                    pen=pg.mkPen(colors['Beta'], width=line_width), 
                    name='β',
                    max_gap=sw_gap_duration,
                    view_box=self.view_boxes[2]
                )
            
            # PANEL 4: V components (Sheath) or Density and Temperature (others)
            if is_sheath:
                # For Sheath: Plot V components
                if sw_data and 'time' in sw_data:
                    times = sw_data['time']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Plot Vy and Vz on left axis
                    if 'Vy' in sw_data:
                        self.plot_with_gaps(
                            self.plots[3], 
                            times_doy, 
                            sw_data['Vy'],
                            pen=pg.mkPen(colors['Vy'], width=line_width), 
                            name='Vy',
                            max_gap=sw_gap_duration
                        )
                    if 'Vz' in sw_data:
                        self.plot_with_gaps(
                            self.plots[3], 
                            times_doy, 
                            sw_data['Vz'],
                            pen=pg.mkPen(colors['Vz'], width=line_width), 
                            name='Vz',
                            max_gap=sw_gap_duration
                        )
                    
                    # Plot Vx on right axis
                    if 'Vx' in sw_data and self.view_boxes[3]:
                        self.plot_with_gaps(
                            None,  # No primary plot
                            times_doy, 
                            sw_data['Vx'],
                            pen=pg.mkPen(colors['Vx'], width=line_width), 
                            name='Vx',
                            max_gap=sw_gap_duration,
                            view_box=self.view_boxes[3]
                        )
            else:
                # For ForbMod/Insitu: Plot Density and Temperature
                if sw_data and 'time' in sw_data:
                    times = sw_data['time']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Plot Density (N) with gap handling
                    if 'N' in sw_data:
                        self.plot_with_gaps(
                            self.plots[3], 
                            times_doy, 
                            sw_data['N'],
                            pen=pg.mkPen(colors['N'], width=line_width), 
                            name='N',
                            max_gap=sw_gap_duration
                        )
                    
                    # Temperature
                    if 'T' in sw_data and self.view_boxes[3]:        
                        self.plot_with_gaps(
                            None,  # No primary plot
                            times_doy, 
                            sw_data['T']/1e5,  # Scale to 10^5 K
                            pen=pg.mkPen(colors['T'], width=line_width), 
                            name='T',
                            max_gap=sw_gap_duration,
                            view_box=self.view_boxes[3]
                        )
                        
                        # Also plot expected temperature if available
                        if 'T_exp' in sw_data:
                            self.plot_with_gaps(
                                None,  # No primary plot
                                times_doy, 
                                sw_data['T_exp']/1e5,  # Scale to 10^5 K
                                pen=pg.mkPen(colors['T_exp'], width=line_width), 
                                name='T_exp',
                                max_gap=sw_gap_duration,
                                view_box=self.view_boxes[3]
                            )
            
            # PANEL 5: Density and Temperature (Sheath) or Primary GCR (others)
            if is_sheath:
                # For Sheath: Plot Density and Temperature
                if sw_data and 'time' in sw_data:
                    times = sw_data['time']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Plot Density (N) with gap handling
                    if 'N' in sw_data:
                        self.plot_with_gaps(
                            self.plots[4], 
                            times_doy, 
                            sw_data['N'],
                            pen=pg.mkPen(colors['N'], width=line_width), 
                            name='N',
                            max_gap=sw_gap_duration
                        )
                    
                    # Temperature
                    if 'T' in sw_data and self.view_boxes[4]:        
                        self.plot_with_gaps(
                            None,  # No primary plot
                            times_doy, 
                            sw_data['T']/1e5,  # Scale to 10^5 K
                            pen=pg.mkPen(colors['T'], width=line_width), 
                            name='T',
                            max_gap=sw_gap_duration,
                            view_box=self.view_boxes[4]
                        )
                        
                        # Also plot expected temperature if available
                        if 'T_exp' in sw_data:
                            self.plot_with_gaps(
                                None,  # No primary plot
                                times_doy, 
                                sw_data['T_exp']/1e5,  # Scale to 10^5 K
                                pen=pg.mkPen(colors['T_exp'], width=line_width), 
                                name='T_exp',
                                max_gap=sw_gap_duration,
                                view_box=self.view_boxes[4]
                            )
            else:
                # For ForbMod/Insitu: Plot Primary GCR
                if 'time' in gcr_data and 'GCR' in gcr_data:
                    times = gcr_data['time']
                    gcr_values = gcr_data['GCR']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Find the first valid value for normalization
                    valid_indices = np.where(~np.isnan(gcr_values))[0]
                    
                    if len(valid_indices) > 0:
                        first_valid_idx = valid_indices[0]
                        first_valid = gcr_values[first_valid_idx]
                        
                        if first_valid != 0:
                            # Calculate normalized values
                            gcr_norm = np.full_like(gcr_values, np.nan)
                            valid_mask = ~np.isnan(gcr_values)
                            gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                            
                            # Plot with proper gap handling
                            self.plot_with_gaps(
                                self.plots[4], 
                                times_doy, 
                                gcr_norm,
                                pen=pg.mkPen(colors['GCR'], width=line_width),
                                name='GCR',
                                symbol='o',
                                symbolSize=3,
                                max_gap=gcr_gap_duration
                            )
            
            # PANEL 6: GCR (primary for Sheath, secondary for others)
            if is_sheath:
                # For Sheath: Plot Primary GCR
                if 'time' in gcr_data and 'GCR' in gcr_data:
                    times = gcr_data['time']
                    gcr_values = gcr_data['GCR']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Find the first valid value for normalization
                    valid_indices = np.where(~np.isnan(gcr_values))[0]
                    
                    if len(valid_indices) > 0:
                        first_valid_idx = valid_indices[0]
                        first_valid = gcr_values[first_valid_idx]
                        
                        if first_valid != 0:
                            # Calculate normalized values
                            gcr_norm = np.full_like(gcr_values, np.nan)
                            valid_mask = ~np.isnan(gcr_values)
                            gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                            
                            # Plot with proper gap handling
                            self.plot_with_gaps(
                                self.plots[5], 
                                times_doy, 
                                gcr_norm,
                                pen=pg.mkPen(colors['GCR'], width=line_width),
                                name='GCR',
                                symbol='o',
                                symbolSize=3,
                                max_gap=gcr_gap_duration
                            )
            else:
                # For ForbMod/Insitu: Plot Secondary GCR
                if 'time' in gcr_secondary_data and 'GCR' in gcr_secondary_data:
                    times = gcr_secondary_data['time']
                    gcr_values = gcr_secondary_data['GCR']
                    
                    # Convert to continuous DOY if spanning multiple years
                    if spans_multiple_years:
                        times_doy = np.array([self.datetime_to_doy(t, continuous_across_years=True, 
                                                                 reference_year=self.reference_year) 
                                             for t in times])
                    else:
                        times_doy = np.array([self.datetime_to_doy(t) for t in times])
                    
                    # Find the first valid value for normalization
                    valid_indices = np.where(~np.isnan(gcr_values))[0]
                    
                    if len(valid_indices) > 0:
                        first_valid_idx = valid_indices[0]
                        first_valid = gcr_values[first_valid_idx]
                        
                        if first_valid != 0:
                            # Calculate normalized values
                            gcr_norm = np.full_like(gcr_values, np.nan)
                            valid_mask = ~np.isnan(gcr_values)
                            gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                            
                            # Plot with proper gap handling
                            self.plot_with_gaps(
                                self.plots[5], 
                                times_doy, 
                                gcr_norm,
                                pen=pg.mkPen(colors['GCR'], width=line_width),
                                name='GCR_secondary',
                                symbol='o',
                                symbolSize=3,
                                max_gap=gcr_gap_duration
                            )
            
            # Update views to ensure proper alignment
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
    
            self.update_datetime_axis()
        
        except Exception as e:
            logger.error(f"Error in _plot_visible_data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    
    def load_data(self):
        """Load and display the data using datetime objects internally"""
        try:
            logger.info("Starting to load data...")
            
            # Load all data for the selected date range
            data_dict = self.data_manager.load_data()
            
            # Check if data is None
            if data_dict is None:
                logger.warning("No data loaded")
                return
            
            # Calculate appropriate x-axis range based on actual data
            start_doy = None
            end_doy = None
            
            # Check if any data spans across year boundary
            spans_years = False
            if 'mf' in data_dict and 'time' in data_dict['mf'] and len(data_dict['mf']['time']) > 0:
                times = data_dict['mf']['time']
                years = set(pd.Timestamp(t).year for t in times)
                spans_years = len(years) > 1
                
                if spans_years:
                    logger.info(f"Data spans multiple years: {sorted(years)}")
                    self.reference_year = min(years)  # Use earliest year as reference
                    
                    # Get the DOY range based on actual data using continuous DOY
                    # Make sure to use the reference_year for conversion 
                    start_doy = self.datetime_to_doy(min(times), continuous_across_years=True, reference_year=self.reference_year)
                    end_doy = self.datetime_to_doy(max(times), continuous_across_years=True, reference_year=self.reference_year)
                else:
                    # Standard DOY calculation (single year)
                    start_doy = self.datetime_to_doy(min(times))
                    end_doy = self.datetime_to_doy(max(times))
            
            # Plot the data
            self.plot_data(data_dict)
            
            # Set appropriate x-axis range after plotting
            if start_doy is not None and end_doy is not None:
                # Add some padding
                padding = (end_doy - start_doy) * 0.05
                for plot in self.plots:
                    plot.setXRange(start_doy - padding, end_doy + padding, padding=0)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def setup_selection_regions(self):
        """Set up selection regions with throttled updates using datetime internally"""
        try:
            self.regions = []
            
            # Use data start date + 0.5 days as initial selection
            initial_start_dt = self.data_manager.start_date + timedelta(days=0.3)
            initial_sheath_dt = initial_start_dt + timedelta(days=1.5)  # Middle point for sheath/MO boundary
            initial_end_dt = initial_start_dt + timedelta(days=2.0)  # Default to 1 day selection
            
            # Convert to DOY for display
            start_doy = self.datetime_to_doy(initial_start_dt)
            sheath_end_doy = self.datetime_to_doy(initial_sheath_dt)
            end_doy = self.datetime_to_doy(initial_end_dt)
            
            # Store the last update time for throttling
            self.last_region_update = time.time()
            self.region_update_delay = 0.1  # seconds
            
            # Initialize labels
            self.start_label = None
            self.end_label = None
            
            # Determine if we're in sheath analysis mode
            is_sheath_analysis = self.analysis_type == "Sheath analysis"
            
            if is_sheath_analysis:
                # Clear any existing regions
                for plot in self.plots:
                    for item in list(plot.items):
                        if isinstance(item, pg.LinearRegionItem):
                            plot.removeItem(item)
                
                # Create regions for each plot
                self.upstream_regions = []
                self.sheath_regions = []
                self.mo_regions = []
                
                # Define initial positions 
                upstream_start_doy = start_doy + 0.3
                upstream_end_doy = start_doy + 1  # Set upstream end to exactly match sheath start
                
                sheath_start_doy = upstream_end_doy 
                sheath_end_doy = sheath_end_doy
                
                mo_start_doy = sheath_end_doy  # MO starts where sheath ends
                mo_end_doy = end_doy
                
                # Create separator position at middle of sheath region
                separator_pos = (sheath_start_doy + sheath_end_doy) / 2
                
                # Initialize separator visibility
                self.separator_visible = False
                if hasattr(self, 'separate_regions_checkbox') and self.separate_regions_checkbox.isChecked():
                    self.separator_visible = True
                
                for plot in self.plots:
                    # 1. Upstream region (yellow)
                    upstream_region = pg.LinearRegionItem(
                        values=(upstream_start_doy, upstream_end_doy),
                        brush=pg.mkBrush(255, 255, 0, 40),  # Yellow
                        movable=True,
                        swapMode='block'
                    )
                    plot.addItem(upstream_region)
                    self.upstream_regions.append(upstream_region)
                    
                    # 2. Sheath region (purple)
                    sheath_region = pg.LinearRegionItem(
                        values=(sheath_start_doy, sheath_end_doy),
                        brush=pg.mkBrush(180, 100, 220, 40),  # Purple
                        movable=True,
                        swapMode='block'
                    )
                    plot.addItem(sheath_region)
                    self.sheath_regions.append(sheath_region)
                    
                    # 3. MO region (blue)
                    mo_region = pg.LinearRegionItem(
                        values=(mo_start_doy, mo_end_doy),
                        brush=pg.mkBrush(100, 149, 237, 40),  # Light blue
                        movable=True,
                        swapMode='block'
                    )
                    plot.addItem(mo_region)
                    self.mo_regions.append(mo_region)
                    
                    # Connect signals - for both finished events and during-drag events
                    upstream_region.sigRegionChangeFinished.connect(self.update_upstream_regions)
                    upstream_region.sigRegionChanged.connect(self.throttled_region_update)
                    
                    sheath_region.sigRegionChangeFinished.connect(self.update_sheath_regions)
                    sheath_region.sigRegionChanged.connect(self.throttled_region_update)
                    
                    mo_region.sigRegionChangeFinished.connect(self.update_mo_regions)
                    mo_region.sigRegionChanged.connect(self.throttled_region_update)
                
                # Create the separator line
                self.create_sheath_separator(separator_pos)
                
                # Store datetime values for internal use
                year = self.data_manager.start_date.year
                self.region_datetime = {
                    'upstream_start': self.data_manager.doy_to_datetime(year, upstream_start_doy),
                    'upstream_end': self.data_manager.doy_to_datetime(year, upstream_end_doy),
                    'sheath_start': self.data_manager.doy_to_datetime(year, sheath_start_doy),
                    'sheath_end': self.data_manager.doy_to_datetime(year, sheath_end_doy),
                    'mo_start': self.data_manager.doy_to_datetime(year, mo_start_doy),
                    'mo_end': self.data_manager.doy_to_datetime(year, mo_end_doy),
                    'separator': self.data_manager.doy_to_datetime(year, separator_pos)
                }
                
                # Make main regions list for compatibility with existing code
                self.regions = self.sheath_regions
                
            else:
                # Standard regions for non-sheath analysis
                for plot in self.plots:
                    region = pg.LinearRegionItem(
                        values=(start_doy, end_doy),
                        brush=pg.mkBrush(100, 100, 255, 50),
                        movable=True
                    )
                    plot.addItem(region)
                    region.sigRegionChangeFinished.connect(self.update_all_regions)
                    region.sigRegionChanged.connect(self.throttled_region_update)
                    self.regions.append(region)
                    
                self.region_datetime = {
                    'start': initial_start_dt,
                    'end': initial_end_dt
                }
                
        except Exception as e:
            logger.error(f"Error setting up selection regions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def update_all_regions(self):
        """Update all regions to match the changed region with throttling, maintaining datetime internally"""
        try:
            sender = self.sender()
            if not sender:
                return
                
            # Don't process if we're already updating or if this is a throttled call
            if hasattr(self, '_updating_regions') and self._updating_regions:
                return
                
            self._updating_regions = True
            
            # Get the region range (in DOY format)
            region_range = sender.getRegion()
            
            # Convert DOY to datetime and store internally
            year = self.data_manager.start_date.year
            self.region_datetime = {
                'start': self.data_manager.doy_to_datetime(year, region_range[0]),
                'end': self.data_manager.doy_to_datetime(year, region_range[1])
            }
            
            # Update all regions except the sender
            for region in self.regions:
                if region != sender:
                    region.blockSignals(True)  # Prevent recursion
                    region.setRegion(region_range)
                    region.blockSignals(False)
                    
            # Update date labels if needed
            self.update_labels(sender)
            
            # Update time display in control panel
            if hasattr(self, 'update_time_display'):
                self.update_time_display()
            
            # Reset flag
            self._updating_regions = False
                    
        except Exception as e:
            logger.error(f"Error updating regions: {str(e)}")
            self._updating_regions = False
    
    def setup_movable_lines(self):
        """Set up movable lines for upstream speed calculation with datetime internally"""
        try:
            # Skip creating the upstream speed lines in sheath analysis mode
            if self.analysis_type == "Sheath analysis":
                return
            
            # Start 0.3 days after the plot start for upstream region
            start_dt = self.data_manager.start_date + timedelta(days=0.3)
            end_dt = start_dt + timedelta(days=0.6)
            
            # Convert to DOY for display
            start_doy = self.datetime_to_doy(start_dt)
            end_doy = self.datetime_to_doy(end_dt)
            
            # Create movable lines with labels
            self.movable_line_start = pg.InfiniteLine(
                pos=start_doy,
                angle=90,
                pen=pg.mkPen('purple', width=2),
                movable=True,
                label='upstream',
                labelOpts={'position': 0.2, 'color': 'purple'}
            )
            
            self.movable_line = self.movable_line_end = pg.InfiniteLine(
                pos=end_doy,
                angle=90,
                pen=pg.mkPen('purple', width=2),
                movable=True,
                label='v',
                labelOpts={'position': 0.2, 'color': 'purple'}
            )
            
            # Create shaded region
            self.upstream_region = pg.LinearRegionItem(
                values=(start_doy, end_doy),
                brush=pg.mkBrush(255, 255, 0, 20),
                movable=False
            )
            
            # Add to velocity plot (index 2)
            self.plots[2].addItem(self.upstream_region)
            self.plots[2].addItem(self.movable_line_start)
            self.plots[2].addItem(self.movable_line_end)
            
            # Connect signals
            self.movable_line_start.sigPositionChanged.connect(self.update_upstream_region)
            self.movable_line_end.sigPositionChanged.connect(self.update_upstream_region)
            
            # Store datetime values for internal use
            self.upstream_datetime = {
                'start': start_dt,
                'end': end_dt
            }
            
        except Exception as e:
            logger.error(f"Error setting up movable lines: {str(e)}")
    
    
    def update_time_display(self):
        """Update time display to show both DOY and datetime format"""
        if not hasattr(self, 'region_start_label') or not hasattr(self, 'region_end_label'):
            return
            
        if not self.regions:
            return
            
        # Get region bounds in DOY
        doy_bounds = self.regions[0].getRegion()
        
        # Use stored datetime values if available, otherwise convert DOY to datetime
        if hasattr(self, 'region_datetime') and 'start' in self.region_datetime and 'end' in self.region_datetime:
            start_date = self.region_datetime['start']
            end_date = self.region_datetime['end']
        else:
            # Convert DOY to datetime for display
            year = self.data_manager.start_date.year
            start_date = self.data_manager.doy_to_datetime(year, doy_bounds[0])
            end_date = self.data_manager.doy_to_datetime(year, doy_bounds[1])
        
        # Update labels
        self.region_start_label.setText(f"Start: DOY {doy_bounds[0]:.2f} ({start_date.strftime('%Y-%m-%d %H:%M')})")
        self.region_end_label.setText(f"End: DOY {doy_bounds[1]:.2f} ({end_date.strftime('%Y-%m-%d %H:%M')})")
    
    def datetime_to_doy(self, dt, continuous_across_years=False, reference_year=None):
        """Convert datetime to day of year, with support for continuous DOY across years"""
        if hasattr(self.data_manager, 'datetime_to_doy'):
            # Use data_manager's enhanced version if available
            return self.data_manager.datetime_to_doy(dt, continuous_across_years, reference_year)
        else:
            # Fallback implementation with continuous DOY support
            if isinstance(dt, datetime):
                # Get standard DOY
                day_of_year = dt.timetuple().tm_yday
                fraction = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
                doy = day_of_year + fraction
                
                # Add days for years after reference_year if continuous mode enabled
                if continuous_across_years and reference_year is not None and dt.year > reference_year:
                    for year in range(reference_year, dt.year):
                        # Add 366 for leap years, 365 for regular years
                        doy += 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
                
                return doy
                
            elif hasattr(dt, 'dtype') and np.issubdtype(dt.dtype, np.datetime64):
                # Convert numpy.datetime64 to Python datetime
                dt_obj = pd.Timestamp(dt).to_pydatetime()
                
                # Get standard DOY
                day_of_year = dt_obj.timetuple().tm_yday
                fraction = (dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second) / 86400.0
                doy = day_of_year + fraction
                
                # Add days for years after reference_year if continuous mode enabled
                if continuous_across_years and reference_year is not None and dt_obj.year > reference_year:
                    for year in range(reference_year, dt_obj.year):
                        # Add 366 for leap years, 365 for regular years
                        doy += 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
                
                return doy
                
            else:
                # If it's already a numeric value (like a DOY), return as is
                if isinstance(dt, (int, float)):
                    return float(dt)
                else:
                    logger.warning(f"Unhandled datetime type in datetime_to_doy: {type(dt)}")
                    return 0.0
    
    def on_calculate_clicked(self):
        """Handle calculate button click event with specific boundary margins"""
        try:
            # Validate selections
            if not self.regions:
                QMessageBox.warning(self, "Warning", "Please define the regions first")
                return
            
            if self.analysis_type == "Sheath analysis":
                # Fixed margin parameter (5 minutes)
                margin_minutes = 5
                margin_delta = timedelta(minutes=margin_minutes)
                
                # Apply margin only to end of upstream and start of sheath
                upstream_start_dt = self.region_datetime['upstream_start']  # No margin
                upstream_end_dt = self.region_datetime['upstream_end'] - margin_delta  # Subtract from end
                sheath_start_dt = self.region_datetime['sheath_start'] + margin_delta  # Add to start
                sheath_end_dt = self.region_datetime['sheath_end']  # No margin
                mo_start_dt = self.region_datetime['mo_start']  # No margin
                mo_end_dt = self.region_datetime['mo_end']  # No margin
                
                # Check if margins create invalid regions
                if upstream_start_dt >= upstream_end_dt or sheath_start_dt >= sheath_end_dt:
                    QMessageBox.warning(self, "Warning", 
                                        "The margin of 5 minutes causes upstream or sheath region to disappear. Please make your regions larger.")
                    return
                
                # Create regions dictionary with selective margins applied
                regions = {
                    'upstream': {
                        'start': upstream_start_dt,
                        'end': upstream_end_dt
                    },
                    'sheath': {
                        'start': sheath_start_dt,
                        'end': sheath_end_dt
                    },
                    'mo': {
                        'start': mo_start_dt,
                        'end': mo_end_dt
                    }
                }
                
                # Handle separator if present - NO MARGINS for separator
                if self.separator_visible and hasattr(self, 'separator_lines'):
                    separator_pos = self.separator_lines[0].value()
                    separator_dt = self.data_manager.doy_to_datetime(self.data_manager.start_date.year, separator_pos)
                    
                    # Only split if separator is within the sheath region
                    if separator_dt > sheath_start_dt and separator_dt < sheath_end_dt:
                        # Add the sub-regions WITHOUT margins at the separator point
                        regions['sheath'] = {
                            'start': sheath_start_dt,
                            'end': separator_dt  # No margin at separator
                        }
                        regions['front_region'] = {
                            'start': separator_dt,  # No margin at separator
                            'end': sheath_end_dt
                        }
                        
                        # Still check if valid regions exist
                        if regions['sheath']['start'] >= regions['sheath']['end'] or regions['front_region']['start'] >= regions['front_region']['end']:
                            QMessageBox.warning(self, "Warning", 
                                               "Invalid sheath regions. Please adjust your separator position.")
                            return
                
                # Calculate for each region
                all_results = []
                
                for region_name, region_data in regions.items():
                    # Create a single-region calculation for each region
                    calc_regions = {
                        'main': region_data,
                        'upstream': regions['upstream']  
                    }
                    
                    calculator = CalculationManager(self.data_manager)
                    calc_results = calculator.perform_calculations(
                        calc_regions,
                        analysis_type=self.analysis_type
                    )
                    
                    if not calc_results:
                        raise ValueError(f"Calculation failed to produce results for {region_name} region")
                        
                    # Add region name for saving
                    calc_results['region'] = region_name
                    all_results.append(calc_results)
            
                # Create publication-quality figure
                fig = self.create_publication_quality_figure(all_results[0], regions)
            
                # Get event date from the calculation results
                doy_start = all_results[0]['timestamps']['doy_start']
                year = self.data_manager.start_date.year
                event_date = self.data_manager.doy_to_datetime(year, doy_start)
                date_str = event_date.strftime('%Y_%m_%d')
                
                try:
                    # Create a single event directory for Sheath analysis
                    sheath_base_dir = os.path.join(
                        self.script_directory,
                        'OUTPUT',
                        self.data_manager.satellite,
                        'Sheath_analysis'
                    )
                    os.makedirs(sheath_base_dir, exist_ok=True)
                    
                    # Create event directory
                    event_dir = os.path.join(sheath_base_dir, date_str)
                    os.makedirs(event_dir, exist_ok=True)

                    lundquist_dir = os.path.join(event_dir, 'lundquist_fit')
                    os.makedirs(lundquist_dir, exist_ok=True)
                    
                    
                    # Create output handler
                    output_handler = OutputHandler(event_dir, self.script_directory)

                    # Save export figure
                    fig_path = output_handler.save_publication_figure(
                        fig, 
                        self.data_manager.satellite,
                        event_date,
                        "Sheath analysis",
                        subtype=None # no subtypes for namin the file
                    )
                    
                    # Save all region results to the CSV
                    for calc_results in all_results:
                        region_name = calc_results['region']
                        
                        # Set analysis type 
                        calc_results['analysis_type'] = "Sheath analysis"
                        
                        output_handler.update_results_csv(
                            sat=self.data_manager.satellite,
                            detector=self.data_manager.detector,
                            observer=self.observer_name,
                            calc_results=calc_results,
                            day=event_date.strftime('%Y/%m/%d'),
                            fit_type = region_name  # Pass region_name instead fit_type parameter
                        )

                    # === HANDLING LUNDQUIST FIT FOR MO REGION ===
                    if 'mo' in regions:
                        # Check if Lundquist fit is enabled
                        lundquist_enabled = True
                        if hasattr(self, 'lundquist_fit_checkbox'):
                            lundquist_enabled = self.lundquist_fit_checkbox.isChecked()

                        if lundquist_enabled:
                            try:
                                # Perform Lundquist fit for MO region in sheath analysis
                                # Import from the lundquist package
                                from lundquist.lundquist_dialog import LundquistParamDialog
                                from lundquist import lundquist_connector
                            
                                # Show dialog to get initial parameters
                                param_dialog = LundquistParamDialog(self)
                                if param_dialog.exec_():
                                    # Get the parameters from the dialog
                                    parameters = param_dialog.get_parameters()
                                    
                                    # Call the connector function with the already-obtained parameters
                                    lundquist_connector.run_lundquist_fit_from_gui(
                                        self,  # parent window
                                        self.data_manager, 
                                        self.region_datetime,
                                        lundquist_dir,  # output directory
                                        parameters  # Add parameters here
                                    )
                            except Exception as e:
                                logger.error(f"Error performing Lundquist fit: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                    # === END OF LUNDQUIST PART ===

                    if not lundquist_enabled: # with fit no need to show this window
                    
                        QMessageBox.information(
                            self,
                            "Success",
                            f"Sheath analysis results saved successfully to {event_dir}"
                        )
                except Exception as e:
                    logger.error(f"Error saving sheath analysis: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Failed to save sheath analysis: {str(e)}")
                    
                plt.close(fig)
            

                
            else:
                # Original calculate logic for non-sheath analysis
                
                # Get datetime values directly from stored values
                region_start_dt = self.region_datetime['start']
                region_end_dt = self.region_datetime['end']
                upstream_start_dt = self.upstream_datetime['start']
                upstream_end_dt = self.upstream_datetime['end']
            
                # Check if upstream window was moved from default position to give the warning
                default_start_dt = self.data_manager.start_date + timedelta(days=0.3)
                default_end_dt = default_start_dt + timedelta(days=0.6)
                
                # Compare with a small time tolerance
                time_tolerance = timedelta(seconds=60)  # 1 minute tolerance
                if (abs(upstream_start_dt - default_start_dt) < time_tolerance and 
                    abs(upstream_end_dt - default_end_dt) < time_tolerance):
                    response = QMessageBox.question(
                        self,
                        "Warning",
                        "The upstream speed (yellow region) appears to be in its default position. "
                        "Are you sure you want to proceed without adjusting it?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if response == QMessageBox.No:
                        return
            
                current_fit_type = self.fit_type.currentText()
            
                # Create regions dictionary
                regions = {
                    'main': {
                        'start': region_start_dt,
                        'end': region_end_dt
                    },
                    'upstream': {
                        'start': upstream_start_dt,
                        'end': upstream_end_dt
                    }
                }
                            
                # calculation for the chosen analysis type
                calculator = CalculationManager(self.data_manager)
                calc_results = calculator.perform_calculations(
                    regions,
                    analysis_type=self.analysis_type
                )
                        
                if not calc_results:
                    raise ValueError("Calculation failed to produce results")
            
                # Create publication-quality figure
                fig = self.create_publication_quality_figure(calc_results)
            
                # Get event date from the calculation results
                doy_start = calc_results['timestamps']['doy_start']
                year = self.data_manager.start_date.year
                event_date = self.data_manager.doy_to_datetime(year, doy_start)
            
                # Process calculation results based on analysis type
                if self.analysis_type == "In-situ analysis":
                    try:
                        # Create insitu_analysis directory for the satellite
                        insitu_dir = os.path.join(
                            self.script_directory,
                            'OUTPUT',
                            self.data_manager.satellite,
                            'insitu_analysis'
                        )
                        os.makedirs(insitu_dir, exist_ok=True)
                        
                        # Create unique ID for this event
                        sat_name = 'nm' if self.data_manager.satellite == 'neutron monitors' else self.data_manager.satellite
                        unique_id = f"Insitu_{sat_name}_{event_date.strftime('%Y_%m_%d')}_{current_fit_type.lower()}"
                        
                        # Add analysis type to calc_results 
                        calc_results['analysis_type'] = "In-situ analysis"  
                        
                        # Create output handler
                        output_handler = OutputHandler(insitu_dir, self.script_directory)
                            
                        # Save export figure
                        fig_path = output_handler.save_publication_figure(
                            fig, 
                            self.data_manager.satellite,
                            event_date,
                            "In-situ analysis",
                            current_fit_type.lower()
                        )
                                            
                        # Update CSV file with results
                        output_handler.update_results_csv(
                            sat=self.data_manager.satellite,
                            detector=self.data_manager.detector,
                            observer=self.observer_name,
                            calc_results=calc_results,
                            day=event_date.strftime('%Y/%m/%d'),
                            fit_type=current_fit_type.lower()
                        )
                        
                    except Exception as e:
                        logger.error(f"Error saving in-situ analysis: {str(e)}")
                        QMessageBox.critical(self, "Error", f"Failed to save in-situ analysis: {str(e)}")
                        
                elif self.analysis_type == "ForbMod":
                    # Handle ForbMod analysis with event folders
                    try:
                        # Create forbmod_analysis/event_date directories
                        forbmod_dir = os.path.join(
                            self.script_directory,
                            'OUTPUT',
                            self.data_manager.satellite,
                            'forbmod_analysis',
                            event_date.strftime('%Y_%m_%d')
                        )
                        os.makedirs(forbmod_dir, exist_ok=True)
                        
                        # Create output handler
                        output_handler = OutputHandler(forbmod_dir, self.script_directory)
                        
                        # Save the publication figure with descriptive name
                        fig_path = output_handler.save_publication_figure(
                            fig, 
                            self.data_manager.satellite,
                            event_date,
                            "ForbMod",
                            current_fit_type.lower()
                        )
                        
                        # Add analysis type to calc_results
                        calc_results['analysis_type'] = "ForbMod"
                        
                        # Create FitWindow with the directory information
                        self.fit_window = FitWindow(
                            sat=self.data_manager.satellite,
                            detector=self.data_manager.detector,
                            observer=self.observer_name,
                            calc_results=calc_results,
                            output_info={
                                'script_directory': self.script_directory,
                                'results_directory': forbmod_dir,
                                'day': event_date.strftime('%Y/%m/%d'),
                                'fit': current_fit_type.lower(),
                                'observer_name': self.observer_name,
                                'data_manager': self.data_manager,
                                'figure': fig
                            },
                            window_manager=self.window_manager
                        )
                        self.fit_window.show()
                        
                    except Exception as e:
                        logger.error(f"Error in ForbMod analysis: {str(e)}")
                        QMessageBox.critical(self, "Error", f"Failed to create ForbMod analysis: {str(e)}")
                    
                plt.close(fig)
                    
        except Exception as e:
            logger.error(f"Error during calculation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")
    
    def setup_control_panel(self, parent_frame):
        """Set up the control panel with datetime-aware labels"""
        # Clear any existing layout
        if parent_frame.layout():
            while parent_frame.layout().count():
                item = parent_frame.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            old_layout = parent_frame.layout()
            parent_frame.setLayout(None)
            old_layout.deleteLater()
        
        # Create new layout
        control_layout = QHBoxLayout(parent_frame)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(4)
        
        # Set minimum height
        parent_frame.setMinimumHeight(50)
        
        # Get DPI factor for consistent scaling
        dpi_factor = 1.0
        if hasattr(self, 'window_manager') and self.window_manager:
            dpi_factor = self.window_manager.settings.get('dpi_factor', 1.0)
        
        # Calculate consistent button dimensions
        button_height = int(35 * dpi_factor)
        nav_button_width = int(100 * dpi_factor)
        square_button_size = int(35 * dpi_factor)
        
        # Get font for buttons
        button_font = self.window_manager.get_sized_font('normal')
        
        # Navigation buttons
        nav_buttons = [
            ("◀ 5 days", lambda: self.adjust_dates(-5)),
            ("◀ 2 days", lambda: self.adjust_dates(-2)),
            ("2 days ▶", lambda: self.adjust_dates(2)),
            ("5 days ▶", lambda: self.adjust_dates(5)),
            ("10 days ▶", lambda: self.adjust_dates(10)),
        ]
        
        for text, callback in nav_buttons:
            btn = QPushButton(text)
            btn.setFixedWidth(nav_button_width)
            btn.setFixedHeight(button_height)
            btn.setFont(button_font)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            control_layout.addWidget(btn)
            control_layout.addSpacing(5)
        
        # Add export button
        export_btn = QPushButton("📥")
        export_btn.setFont(button_font)
        export_btn.clicked.connect(self.export_data)
        export_btn.setToolTip("Export Data")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #B8D2FF;  /* Light blue-purple color */
                color: #2C3E50;  /* Dark text for contrast */
                border: none;
                border-radius: 3px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9BB8F0;
            }
        """)
        
        # Set consistent dimensions for export button
        export_btn.setFixedWidth(square_button_size)
        export_btn.setFixedHeight(button_height)
        font = export_btn.font()
        font.setPointSizeF(font.pointSizeF() * 1.2)  # Slightly larger for the icon
        export_btn.setFont(font)
        
        control_layout.addWidget(export_btn)
        
        # Add stretch to push remaining controls to the right
        control_layout.addStretch()
        
        # Add controls based on analysis type
        if self.analysis_type == "Sheath analysis":
            # For Sheath analysis: Use simplified approach with combined labels
            
            # Front region checkbox (with "drag here" in the text)
            self.separate_regions_checkbox = QCheckBox("Add front region")
            if hasattr(self, 'window_manager') and self.window_manager:
                self.separate_regions_checkbox.setFont(button_font)
            self.separate_regions_checkbox.setChecked(False)
            self.separate_regions_checkbox.toggled.connect(self.toggle_separator_visibility)
            
            # Lundquist fit checkbox
            self.lundquist_fit_checkbox = QCheckBox("Use Lundquist fit")
            if hasattr(self, 'window_manager') and self.window_manager:
                self.lundquist_fit_checkbox.setFont(button_font)
            self.lundquist_fit_checkbox.setChecked(True)  # Enabled by default
            self.lundquist_fit_checkbox.setToolTip("Perform Lundquist fit for MO region")
            
            # Add checkboxes to control layout
            control_layout.addWidget(self.separate_regions_checkbox)
            control_layout.addSpacing(10)
            control_layout.addWidget(self.lundquist_fit_checkbox)
        else:
            # Keep the original fit type dropdown for other analysis types
            self.fit_type = QComboBox()
            if self.analysis_type == "In-situ analysis":
                self.fit_type.addItems(["Test", "Inner", "Extended", "Optimal", "Sheath", "Magnetic_Cloud"])
            else:
                self.fit_type.addItems(["Test", "Inner", "Extended", "Optimal"])
                
            self.fit_type.setFixedWidth(int(120 * dpi_factor))
            self.fit_type.setFixedHeight(button_height)
            self.fit_type.setFont(button_font)
            self.fit_type.setStyleSheet("""
                QComboBox {
                    border: 1px solid #BDBDBD;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: white;
                    color: black;
                }
                QComboBox:hover {
                    border: 1px solid #2196F3;
                    background-color: #E3F2FD;
                }
                QComboBox QAbstractItemView {
                    border: 1px solid #BDBDBD;
                    background-color: white;
                    color: black;
                    selection-background-color: #2196F3;
                    selection-color: white;
                }
            """)
            control_layout.addWidget(self.fit_type)
        
        control_layout.addSpacing(10)
        
        # Calculate button
        if self.analysis_type == "In-situ analysis" or self.analysis_type == "Sheath analysis":
            calc_btn = QPushButton("Save")
        else:
            calc_btn = QPushButton("Calculate")
        
        calc_btn.setFixedWidth(int(100 * dpi_factor))
        calc_btn.setFixedHeight(button_height)
        calc_btn.setFont(button_font)
        calc_btn.clicked.connect(self.on_calculate_clicked)
        calc_btn.setToolTip("Calculate parameters and save to a file")
        calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a365d;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2c4c8c;  
            }
        """)
        control_layout.addWidget(calc_btn)
        
        # Connect region change to update time display
        for region in self.regions:
            region.sigRegionChangeFinished.connect(self.update_time_display)
            
        # Initial update of time display
        self.update_time_display()

    
    def on_fit_type_changed(self, new_fit_type):
        """Handle fit type dropdown changes"""
        try:
            # Check if we need to switch to sheath analysis mode
            if new_fit_type == "Sheath" and self.analysis_type == "In-situ analysis":
                # Recreate control panel with checkbox
                self.setup_control_panel(self.findChild(QFrame))
                
            elif hasattr(self, 'separate_regions_checkbox') and self.separate_regions_checkbox:
                # We were in sheath mode but switched to something else
                # Recreate control panel with dropdown
                self.setup_control_panel(self.findChild(QFrame))
            
        except Exception as e:
            logger.error(f"Error handling fit type change: {str(e)}")

    
    def update_upstream_region(self):
        """Update upstream region when movable lines change position"""
        try:
            if self.movable_line_start and self.movable_line_end:
                # Get line positions
                start_pos = self.movable_line_start.value()
                end_pos = self.movable_line_end.value()
                
                # Update shaded region
                self.upstream_region.setRegion((min(start_pos, end_pos), max(start_pos, end_pos)))
                
                # Update datetime values
                year = self.data_manager.start_date.year
                self.upstream_datetime = {
                    'start': self.data_manager.doy_to_datetime(year, start_pos),
                    'end': self.data_manager.doy_to_datetime(year, end_pos)
                }
        except Exception as e:
            logger.error(f"Error updating upstream region: {str(e)}")

    def adjust_dates(self, days):
        """Handle date adjustment with datetime objects across year boundaries"""
        try:
            # Get current dates
            new_start = self.data_manager.start_date + timedelta(days=days)
            new_end = self.data_manager.end_date + timedelta(days=days)
            
            # Save window geometry before navigation
            if self.window_manager:
                self.window_manager.save_window_geometry(self, 'plot_window')
                
            # Clear old references to year-specific attributes
            if hasattr(self, 'reference_year'):
                delattr(self, 'reference_year')
            
            # Important: Clear plots before changing dates to remove old data
            self.clear_plots()
            
            # Update data in data_manager with new date range
            self.on_dates_changed(new_start, new_end)
            
            # Update parent window dates if it exists
            if self.parent:
                if hasattr(self.parent, 'start_date_input'):
                    self.parent.start_date_input.setText(new_start.strftime("%Y/%m/%d"))
                if hasattr(self.parent, 'end_date_input'):
                    self.parent.end_date_input.setText(new_end.strftime("%Y/%m/%d"))
                    
                # Update settings
                if hasattr(self.parent, 'settings_manager'):
                    self.parent.settings_manager.set_last_dates(new_start, new_end)
            
            # After data is loaded, ensure view boxes are properly updated
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
                    
        except Exception as e:
            logger.error(f"Error adjusting dates: {str(e)}")
            QMessageBox.warning(self, "Warning", "Failed to adjust dates")

    
    def setup_plots(self, parent_layout):
        """Set up plots with different panel configurations based on analysis type"""
        try:
            plots_frame = QFrame()
            plots_frame.setFrameShape(QFrame.StyledPanel)
            plots_layout = QVBoxLayout(plots_frame)
            plots_layout.setSpacing(1)
            plots_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            self.plots = []
            self.view_boxes = []
            
            # Calculate DOY range to ensure full range is visible
            start_doy = self.datetime_to_doy(self.data_manager.start_date)
            end_doy = self.datetime_to_doy(self.data_manager.end_date)
            
            # Always use 6 panels but with different configurations
            num_panels = 6
            
            def make_update_function(plot_item, view_box_item):
                def update():
                    if view_box_item is not None:
                        view_box_item.setGeometry(plot_item.vb.sceneBoundingRect())
                return update
            
            # Create panels based on analysis type
            if self.analysis_type == "Sheath analysis":
                # Panel layout for Sheath analysis:
                # 1. B and dB
                # 2. B components
                # 3. V and Beta
                # 4. V components
                # 5. Density and Temperature
                # 6. GCR (primary)
                panels_with_viewbox = [0, 2, 3, 4]  # Panels with secondary y-axis
            else:
                # Panel layout for ForbMod/Insitu:
                # 1. B and dB
                # 2. B components
                # 3. V and Beta
                # 4. Density and Temperature
                # 5. Primary GCR
                # 6. Secondary GCR
                panels_with_viewbox = [0, 2, 3]  # Panels with secondary y-axis
            
            # Create all panels
            for i in range(num_panels):
                plot_container = QWidget()
                container_layout = QVBoxLayout(plot_container)
                container_layout.setContentsMargins(1, 1, 1, 1)
                
                pw = pg.PlotWidget()
                pw.setBackground('w')
                pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                pw.setMinimumHeight(100)
                
                container_layout.addWidget(pw)
                plots_layout.addWidget(plot_container)
            
                plot = pw.plotItem
                plot.showAxis('right')
                view_box = None
            
                # Add view box only for panels that need secondary y-axis
                if i in panels_with_viewbox:
                    view_box = pg.ViewBox()
                    plot.scene().addItem(view_box)
                    plot.getAxis('right').linkToView(view_box)
                    view_box.setXLink(plot.vb)
                    
                    # Connect update function
                    plot.vb.sigResized.connect(make_update_function(plot, view_box))


                    # Special handling for beta plot
                    if i == 2:  # Beta plot
                        try:
                            # Set fixed range - linear mode with range 0-2
                            view_box.setYRange(-0.2, 2.2, padding = 0)
                            view_box.disableAutoRange()
                            
                            # Set specific label
                            plot.getAxis('right').setLabel('β')
                            
                            # Connect update function - Modified to safely check for None
                            def update_view():
                                if view_box is not None and plot is not None and hasattr(plot, 'vb'):
                                    view_box.setGeometry(plot.vb.sceneBoundingRect())
                                    view_box.setYRange(-0.2, 2.2, padding=0)
                            
                            plot.vb.sigResized.connect(update_view)
                            
                        except Exception as e:
                            # Fallback if there's an error
                            print(f"Beta axis setup error: {e}")

                
                else:
                    # Hide values on right axis for panels without secondary y-axis
                    plot.getAxis('right').setStyle(showValues=False)
                
                # Set up left and right axis widths
                plot.getAxis('left').setWidth(20)
                plot.getAxis('right').setWidth(20)
            
                pw.showGrid(x=False, y=False)
                
                if i > 0:
                    pw.setXLink(self.plots[0].getViewBox())
                    
                self.plots.append(plot)
                self.view_boxes.append(view_box)
            
            # Set initial X range 
            x_padding = (end_doy - start_doy) * 0.05  # 5% padding on each side
            self.plots[0].setXRange(start_doy - x_padding, end_doy + x_padding, padding=0)

            # Disable autoscale for units in y axis
            for plot in self.plots:
                plot.getAxis('left').enableAutoSIPrefix(False)
                plot.getAxis('right').enableAutoSIPrefix(False)
            
            parent_layout.addWidget(plots_frame)
            
            # Store the analysis type for reference in other methods
            self.is_sheath_analysis = (self.analysis_type == "Sheath analysis")
            
            # Set up all other components
            self.setup_plot_labels()
            self.setup_selection_regions()
            self.setup_helper_lines()
            self.setup_movable_lines()
            self.apply_scaling_to_plot_widgets()
    
            # Connect the update function
            self.plots[num_panels-1].getViewBox().sigRangeChanged.connect(self.update_datetime_axis)
            # Add initial call to ensure labels appear on startup
            self.update_datetime_axis()
    
        except Exception as e:
            logger.error(f"Error setting up plots: {str(e)}")
            raise

    
    # Helper function to adjust panel index based on analysis type
    def get_adjusted_panel_index(self, base_index):
        """Convert standard panel index to actual index, accounting for extra V component panel in sheath mode"""
        if not hasattr(self, 'has_v_components_panel') or not self.has_v_components_panel:
            return base_index
            
        # For sheath analysis, we have an extra panel at index 3
        # so panels 3+ need to be shifted by 1
        if base_index >= 3:
            return base_index + 1
        return base_index
    


    def create_publication_quality_figure(self, calc_results, regions=None):
        """Create a publication-quality figure for any analysis type"""
        try:
            # Determine number of panels based on analysis type
            # For sheath analysis 6 panels (5 IP + 1 GCR)
            # For ForbMod/Insitu 6 panels (4 IP + 1 GCR + 1 additional GCR)
            num_panels = 6
            
            fig = plt.figure(figsize=(14, 16))
            plt.style.use('default')
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'sans-serif',
                'mathtext.default': 'regular'
            })
    
            # Determine if we're working with the sheath analysis multi-region mode
            is_sheath_mode = self.analysis_type == "Sheath analysis" and regions is not None
    
            # Get view range (as DOY for plotting)
            view_range = self.plots[0].getViewBox().viewRange()
            x_min, x_max = view_range[0]
            padding = (x_max - x_min) * 0.02
            x_min -= padding
            x_max += padding
            
            # Setup gridspec with appropriate number of panels
            gs = fig.add_gridspec(num_panels, 1, height_ratios=[1] * num_panels, hspace=0.15)
    
            # Colors dictionary
            colors = {
                'B': '#000000',      # Black (mag field)
                'dB': '#808080',     # Gray
                'Bx': '#FF0000',     # Red
                'By': '#0000FF',     # Blue
                'Bz': '#006400',     # Dark Green
                'V': '#000000',      # Black (velocity)
                'Vx': '#FF0000',     # Red - velocity components
                'Vy': '#0000FF',     # Blue
                'Vz': '#006400',     # Dark Green
                'Beta': '#0000FF',   # Blue
                'T': '#FF0000',      # Red
                'N': '#000000',      # Black (density)
                'GCR': '#000000'     # Black
            }
    
            # Get data from current_data
            mf_data = self.current_data.get('mf', {})
            sw_data = self.current_data.get('sw', {})
            gcr_data = self.current_data.get('gcr', {})
            gcr_secondary_data = self.current_data.get('gcr_secondary', {})
    
            for i in range(num_panels):
                ax = fig.add_subplot(gs[i])
                ax2 = ax.twinx() if i in [0, 2, 3, 4, 5,6] else None
    
                # Panel 1: B and dB
                if i == 0:
                    if 'B' in mf_data and 'time' in mf_data:
                        # Convert times to DOY
                        times_doy = np.array([self.datetime_to_doy(t) for t in mf_data['time']])
                        ax.plot(times_doy, mf_data['B'], color=colors['B'], label='B')
                    ax.set_ylabel('B [nT]', color='black')
                    ax.tick_params(axis='y', labelcolor='black')
                    
                    if ax2 and 'dB' in mf_data and 'time' in mf_data:
                        times_doy = np.array([self.datetime_to_doy(t) for t in mf_data['time']])
                        ax2.plot(times_doy, mf_data['dB'], color=colors['dB'], label='dB')
                        ax2.set_ylabel('dB [nT]', color='gray')
                        ax2.tick_params(axis='y', labelcolor='gray')
                        ax2.spines['right'].set_color('gray')
                
                # Panel 2: B components
                elif i == 1:
                    if 'time' in mf_data:
                        times_doy = np.array([self.datetime_to_doy(t) for t in mf_data['time']])
                        
                        if 'Bx' in mf_data:
                            ax.plot(times_doy, mf_data['Bx'], color=colors['Bx'], label='Bx')
                        if 'By' in mf_data:
                            ax.plot(times_doy, mf_data['By'], color=colors['By'], label='By')
                        if 'Bz' in mf_data:
                            ax.plot(times_doy, mf_data['Bz'], color=colors['Bz'], label='Bz')
                    
                    ax.set_ylabel('B [nT]', color='black')
                    ax.tick_params(axis='y', labelcolor='black')
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.legend(loc='upper right')
                
                # Panel 3: V and Beta or V components (for sheath analysis)
                elif i == 2:  # V components panel
                    if self.analysis_type == "Sheath analysis":
                        # For sheath analysis: Plot Vy, Vz on left and Vx on right
                        if 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            
                            # Plot Vy and Vz on left axis
                            if 'Vy' in sw_data:
                                ax.plot(times_doy, sw_data['Vy'], color='#0000FF', label='Vy')  # Blue
                            if 'Vz' in sw_data:
                                ax.plot(times_doy, sw_data['Vz'], color='#006400', label='Vz')  # Dark Green
                                
                            # Add Vx on right axis - exactly like temperature
                            if 'Vx' in sw_data:
                                # Create a twin axis for Vx
                                #ax2 = ax.twinx()
                                
                                # Plot and configure - identical to temperature approach
                                ax2.plot(times_doy, sw_data['Vx'], color='#FF0000', label='Vx')
                                ax2.set_ylabel('Vx [km/s]', color='#FF0000')
                                ax2.tick_params(axis='y', labelcolor='#FF0000')
                                ax2.spines['right'].set_color('#FF0000')
                                
                                # Add legend - exactly as done for temperature
                                lines1, labels1 = ax.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        
                        ax.set_ylabel('Vy, Vz [km/s]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    else:
                        # For other analysis types, use original V + Beta panel
                        if 'V' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            ax.plot(times_doy, sw_data['V'], color=colors['V'], label='V', zorder=5)  # Higher zorder to ensure V is on top
                        ax.set_ylabel('V [km/s]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        if ax2 and 'Beta' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            # Plot Beta with thinner line
                            ax2.plot(times_doy, sw_data['Beta'], color=colors['Beta'], linewidth=1, label='β', zorder=4) # beta curve below V + thinner
                            ax2.set_ylabel('β', color='blue')
                            ax2.tick_params(axis='y', labelcolor='blue')
                            ax2.spines['right'].set_color('blue')
                            # Set linear scale with horizontal line at β=1
                            ax2.set_yscale('linear')
                            ax2.set_ylim(0, 2)
                            ax2.axhline(y=1, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
                
                # Panel 4: V and Beta for sheath analysis or Density and Temperature for others
                elif i == 3:
                    if self.analysis_type == "Sheath analysis":
                        # For sheath analysis: Show V + Beta here
                        if 'V' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            ax.plot(times_doy, sw_data['V'], color=colors['V'], label='V', zorder=5)  # Higher zorder to ensure V is on top
                        ax.set_ylabel('V [km/s]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        if ax2 and 'Beta' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            # Plot Beta with thinner line
                            ax2.plot(times_doy, sw_data['Beta'], color=colors['Beta'], linewidth=0.8, label='β', zorder=4)
                            ax2.set_ylabel('β', color='blue')
                            ax2.tick_params(axis='y', labelcolor='blue')
                            ax2.spines['right'].set_color('blue')
                            # Set linear scale with horizontal line at β=1
                            ax2.set_yscale('linear')
                            ax2.set_ylim(0, 2)
                            ax2.axhline(y=1, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
                    else:
                        # For other analysis types: Show density and temperature
                        if 'N' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            ax.plot(times_doy, sw_data['N'], color=colors['N'], label='N')
                        ax.set_ylabel('n [cm⁻³]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        if ax2 and 'T' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            # Convert to 10⁵ K
                            ax2.plot(times_doy, sw_data['T']/1e5, color=colors['T'], label='T')
                            ax2.set_ylabel('T [10⁵ K]', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                            ax2.spines['right'].set_color('red')
                        
                        # Add expected temperature
                        if ax2 and 'T_exp' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            ax2.plot(times_doy, sw_data['T_exp']/1e5, color='blue', label='T_exp')
                            ax2.legend(loc='upper right')
                
                # Panel 5/4: Density and Temperature for sheath or Primary GCR for others
                elif i == 4:
                    if self.analysis_type == "Sheath analysis":
                        
                        # For sheath analysis: Density and Temperature
                        if 'N' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])
                            ax.plot(times_doy, sw_data['N'], color=colors['N'], label='N')
                        ax.set_ylabel('n [cm⁻³]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        if 'T' in sw_data and 'time' in sw_data:
                            times_doy = np.array([self.datetime_to_doy(t) for t in sw_data['time']])

                            # Convert to 10⁵ K
                            ax2.plot(times_doy, sw_data['T']/1e5, color=colors['T'], label='T')
                            ax2.set_ylabel('T [10⁵ K]', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                            ax2.spines['right'].set_color('red')
                        
                            # Add expected temperature
                            if 'T_exp' in sw_data:
                                ax2.plot(times_doy, sw_data['T_exp']/1e5, color='blue',  label='T_exp')
                                ax2.legend(loc='upper right')
                    else:
                        # For other analysis types: Primary GCR
                        if 'time' in gcr_data and 'GCR' in gcr_data:
                            # Normalize GCR
                            gcr_values = gcr_data['GCR']
                            times = gcr_data['time']
                            times_doy = np.array([self.datetime_to_doy(t) for t in times])
                            
                            valid_indices = np.where(~np.isnan(gcr_values))[0]
                            
                            if len(valid_indices) > 0:
                                first_valid_idx = valid_indices[0]
                                first_valid = gcr_values[first_valid_idx]
                                
                                if first_valid != 0:
                                    gcr_norm = np.full_like(gcr_values, np.nan)
                                    valid_mask = ~np.isnan(gcr_values)
                                    gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                                    
                                    ax.plot(times_doy, gcr_norm, color=colors['GCR'],  label='GCR')
                        
                        ax.set_ylabel('GCR [%]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        # Add detector name
                        if self.data_manager.satellite in self.data_manager.detector:
                            detector_name = self.data_manager.detector[self.data_manager.satellite]
                            ax_twin = ax2#ax.twinx()
                            ax_twin.set_ylabel(detector_name, color='black')
                            ax_twin.tick_params(axis='y', labelcolor='black', labelleft=False, labelright=False)
                            ax_twin.set_yticks([])
                
                # Panel 6/5: GCR for both analysis types
                elif i == 5:
                    if self.analysis_type == "Sheath analysis":
                        # For sheath analysis: Primary GCR
                        if 'time' in gcr_data and 'GCR' in gcr_data:
                            # Normalize GCR
                            gcr_values = gcr_data['GCR']
                            times = gcr_data['time']
                            times_doy = np.array([self.datetime_to_doy(t) for t in times])
                            
                            valid_indices = np.where(~np.isnan(gcr_values))[0]
                            
                            if len(valid_indices) > 0:
                                first_valid_idx = valid_indices[0]
                                first_valid = gcr_values[first_valid_idx]
                                
                                if first_valid != 0:
                                    gcr_norm = np.full_like(gcr_values, np.nan)
                                    valid_mask = ~np.isnan(gcr_values)
                                    gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                                    
                                    ax.plot(times_doy, gcr_norm, color=colors['GCR'], label='GCR')
                        
                        ax.set_ylabel('GCR [%]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        # Add detector name to right axis instead of text in plot
                        if self.data_manager.satellite in self.data_manager.detector:
                            detector_name = self.data_manager.detector[self.data_manager.satellite]
                            ax_twin = ax2
                            ax_twin.set_ylabel(detector_name, color='black')
                            ax_twin.tick_params(axis='y', labelcolor='black', labelleft=False, labelright=False)
                            ax_twin.set_yticks([]) 
                    else:
                        # For other analysis types: Secondary GCR
                        if 'time' in gcr_secondary_data and 'GCR' in gcr_secondary_data:
                            # Normalize GCR
                            gcr_values = gcr_secondary_data['GCR']
                            times = gcr_secondary_data['time']
                            times_doy = np.array([self.datetime_to_doy(t) for t in times])
                            
                            valid_indices = np.where(~np.isnan(gcr_values))[0]
                            
                            if len(valid_indices) > 0:
                                first_valid_idx = valid_indices[0]
                                first_valid = gcr_values[first_valid_idx]
                                
                                if first_valid != 0:
                                    gcr_norm = np.full_like(gcr_values, np.nan)
                                    valid_mask = ~np.isnan(gcr_values)
                                    gcr_norm[valid_mask] = (gcr_values[valid_mask] - first_valid) / first_valid * 100
                                    
                                    ax.plot(times_doy, gcr_norm, color=colors['GCR'],  label='GCR')
                        
                        ax.set_ylabel('GCR [%]', color='black')
                        ax.tick_params(axis='y', labelcolor='black')
                        
                        # Add secondary detector name
                        if self.data_manager.satellite in self.data_manager.secondary_detector:
                            detector_name = self.data_manager.secondary_detector.get(self.data_manager.satellite, "")
                            ax_twin = ax2
                            ax_twin.set_ylabel(detector_name, color='black')
                            ax_twin.tick_params(axis='y', labelcolor='black', labelleft=False, labelright=False)
                            ax_twin.set_yticks([]) 
                            
                #########
                # Add region shading based on analysis type
                if is_sheath_mode:
                    # Sheath analysis with multiple regions
                    # Create a flag to determine if we have a split sheath
                    has_split_sheath = 'front_region' in regions
                    
                    # Upstream region (yellow)
                    upstream_doy_start = self.datetime_to_doy(regions['upstream']['start'])
                    upstream_doy_end = self.datetime_to_doy(regions['upstream']['end'])
                    ax.axvspan(upstream_doy_start, upstream_doy_end, alpha=0.13, color='yellow', label='Upstream' if i == 1 else "")
                    
                    # Sheath region (lighter purple with slightly more opacity)
                    sheath_doy_start = self.datetime_to_doy(regions['sheath']['start'])
                    sheath_doy_end = self.datetime_to_doy(regions['sheath']['end'])
                    
                    if has_split_sheath:
                        front_doy_start = self.datetime_to_doy(regions['front_region']['start'])
                        front_doy_end = self.datetime_to_doy(regions['front_region']['end'])
                        
                        # First part of sheath (more visible purple)
                        ax.axvspan(sheath_doy_start, front_doy_start, alpha=0.3, color='mediumpurple', 
                                  label='Sheath' if i == 1 else "")
                        
                        # Front region (darker with more opacity to show transition)
                        # Using a blend of purple and blue 
                        ax.axvspan(front_doy_start, front_doy_end, alpha=0.25, color='slateblue', 
                                  label='Front region' if i == 1 else "")
                        
                        # Add vertical purple separator as requested
                        ax.axvline(x=front_doy_start, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
                    else:
                        # Normal sheath (one region more visible purple)
                        ax.axvspan(sheath_doy_start, sheath_doy_end, alpha=0.3, color='mediumpurple', 
                                  label='Sheath' if i == 1 else "")
                    
                    # MO region 
                    mo_doy_start = self.datetime_to_doy(regions['mo']['start'])
                    mo_doy_end = self.datetime_to_doy(regions['mo']['end'])
                    ax.axvspan(mo_doy_start, mo_doy_end, alpha=0.2, color='blue', 
                              label='MO' if i == 1 else "")
                    
                    # Add vertical lines at boundaries 
                    #ax.axvline(x=upstream_doy_start, color='black', linestyle='-', alpha=0.3)
                    #ax.axvline(x=upstream_doy_end, color='black', linestyle='-', alpha=0.3)
                    #ax.axvline(x=mo_doy_end, color='black', linestyle='-', alpha=0.3)
                    if not has_split_sheath:
                        ax.axvline(x=sheath_doy_end, color='black', linestyle='-', alpha=0.3)
                else:
                    # Standard plotting with main region + upstream speed
                    # Main region
                    doy_start = calc_results['timestamps']['doy_start']
                    doy_end = calc_results['timestamps']['doy_end']
                    ax.axvspan(doy_start, doy_end, alpha=0.2, color='blue', label='ICME' if i == 1 else "")
                    
                    # Add vertical lines at boundaries
                    #ax.axvline(x=doy_start, color='black', linestyle='-', alpha=0.5)
                    #ax.axvline(x=doy_end, color='black', linestyle='-', alpha=0.5)
                    

                # Set axis limits
                ax.set_xlim(x_min, x_max)
                #to plot grid
                #ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
                # Hide x-labels for all plots except the bottom one
                if i < num_panels - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    # Only add DOY labels to bottom axis
                    ax.set_xlabel(f'DOY [{self.data_manager.start_date.year}]')
    
                # Add title to the top plot
                if i == 0:
                    title = f"{self.data_manager.satellite} {self.data_manager.start_date.strftime('%Y/%m/%d')}"
                    if is_sheath_mode:
                        title += " - Sheath Analysis"
                    elif self.analysis_type == "In-situ analysis":
                        title += " - In-situ Analysis"
                    ax.set_title(title, pad=20)
    
            # Add legend based on analysis type
            if is_sheath_mode:
                # Legend for sheath mode
                ax_legend = fig.add_subplot(111)
                ax_legend.axis('off')
                
                # Create custom legend patches
                from matplotlib.patches import Patch
                
                # Check if we have a split sheath region
                if 'front_region' in regions:
                    legend_elements = [
                        Patch(facecolor='yellow', alpha=0.2, label='Upstream'),
                        Patch(facecolor='mediumpurple', alpha=0.3, label='Sheath'),
                        Patch(facecolor='slateblue', alpha=0.25, label='Front region'),
                        Patch(facecolor='blue', alpha=0.2, label='MO')
                    ]
                else:
                    legend_elements = [
                        Patch(facecolor='yellow', alpha=0.2, label='Upstream'),
                        Patch(facecolor='mediumpurple', alpha=0.3, label='Sheath'),
                        Patch(facecolor='blue', alpha=0.2, label='MO')
                    ]
                
                # Place the legend at the top of the figure
                ax_legend.legend(handles=legend_elements, loc='upper center', 
                                 bbox_to_anchor=(0.5, 1.02), ncol=4 if 'front_region' in regions else 3, 
                                 frameon=False)
            fig.set_tight_layout(False)
            fig.subplots_adjust(hspace=0.15, left=0.1, right=0.9, top=0.95, bottom=0.05)
            return fig
    
        except Exception as e:
            logger.error(f"Error creating publication figure: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating figure: {str(e)}", ha='center', va='center')
            return fig
    
    def clear_plots(self, keep_regions=False):
        """Clear all plots including view boxes, optionally keeping regions"""
        try:
            # Save regions if needed
            saved_regions = None
            if keep_regions and self.regions:
                saved_regions = [region.getRegion() for region in self.regions]
            
            # Clear all plots
            for plot in self.plots:
                # Keep only the regions if requested
                if keep_regions:
                    # Get all items that aren't LinearRegionItems or InfiniteLines
                    items_to_remove = []
                    for item in plot.items:
                        if not isinstance(item, (pg.LinearRegionItem, pg.InfiniteLine)):
                            items_to_remove.append(item)
                    
                    # Remove non-region items
                    for item in items_to_remove:
                        plot.removeItem(item)
                else:
                    plot.clear()
                    
            # Clear view boxes
            for view_box in self.view_boxes:
                if view_box is not None:
                    items_to_remove = list(view_box.addedItems)
                    for item in items_to_remove:
                        view_box.removeItem(item)
            
            # If we're not keeping regions, restore them
            if not keep_regions:
                self.setup_helper_lines()
                self.setup_selection_regions()
                self.setup_movable_lines()
            elif saved_regions:
                # Restore region positions
                for i, region in enumerate(self.regions):
                    if i < len(saved_regions):
                        region.setRegion(saved_regions[i])
        
        except Exception as e:
            logger.error(f"Error clearing plots: {str(e)}")

    def update_datetime_axis(self):
        """Add datetime labels to the x-axis of the bottom plot with support for multi-year data"""
        try:
            if len(self.plots) < 6:
                return
                    
            bottom_plot = self.plots[5]
            bottom_axis = bottom_plot.getAxis('bottom')
            
            # Get current view range (in DOY format)
            x_range = bottom_plot.getViewBox().viewRange()[0]
            start_doy, end_doy = x_range
            
            # Ensure the range is valid
            if math.isnan(start_doy) or math.isnan(end_doy) or start_doy >= end_doy:
                return
            
            # Check if we're in multi-year mode
            spans_multiple_years = hasattr(self, 'reference_year')
            base_year = self.reference_year if spans_multiple_years else self.data_manager.start_date.year
                        
            # Convert DOY to datetime objects, handling continuous DOY values
            if spans_multiple_years:
                start_date = self.data_manager.doy_to_datetime(base_year, start_doy, continuous_across_years=True)
                end_date = self.data_manager.doy_to_datetime(base_year, end_doy, continuous_across_years=True)
            else:
                start_date = self.data_manager.doy_to_datetime(base_year, start_doy)
                end_date = self.data_manager.doy_to_datetime(base_year, end_doy)
                    
            # Calculate date span
            date_span = (end_date - start_date).total_seconds() / 86400  # Convert to days
                    
            # Create tick marks with dates
            ticks = []
                    
            # Always include start and end dates
            ticks.append((start_doy, start_date.strftime('%Y-%m-%d %H:%M')))
            ticks.append((end_doy, end_date.strftime('%Y-%m-%d %H:%M')))
                    
            # Add intermediate ticks based on span duration
            if date_span > 1:
                # For spans longer than a day, add daily markers
                current_date = start_date.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                while current_date < end_date:
                    # Convert datetime to DOY (continuous if multi-year)
                    if spans_multiple_years:
                        current_doy = self.datetime_to_doy(current_date, continuous_across_years=True, 
                                                          reference_year=base_year)
                    else:
                        current_doy = self.datetime_to_doy(current_date)
                        
                    # Show year in date format for multi-year data
                    date_format = '%Y-%m-%d' 
                        
                    # Add tick with date
                    ticks.append((current_doy, current_date.strftime(date_format)))
                    # Move to next day
                    current_date += timedelta(days=1)
            else:
                # For shorter spans, add hourly markers
                current_date = start_date.replace(minute=0, second=0) + timedelta(hours=1)
                while current_date < end_date:
                    # Convert datetime to DOY (continuous if multi-year)
                    if spans_multiple_years:
                        current_doy = self.datetime_to_doy(current_date, continuous_across_years=True, 
                                                          reference_year=base_year)
                    else:
                        current_doy = self.datetime_to_doy(current_date)
                        
                    # Add tick with time only
                    ticks.append((current_doy, current_date.strftime('%H:%M')))
                    # Move to next hour
                    current_date += timedelta(hours=1)
                    
            # Sort ticks by position
            ticks.sort(key=lambda x: x[0])
                    
            # Set the ticks on the axis
            bottom_axis.setTicks([ticks, []])
            bottom_axis.showLabel(True)
            bottom_axis.setStyle(showValues=True)
            
            # If we're in multi-year mode, update the axis label to show range of years
            if spans_multiple_years:
                # Get the years in the view range
                start_year = start_date.year
                end_year = end_date.year
                
                # Update the axis label to show year range
                if start_year == end_year:
                    x_label = f'DOY [{start_year}]'
                else:
                    x_label = f'Continuous DOY [{start_year}-{end_year}]'
                    
                # Set the updated label
                for plot in self.plots:
                    if plot == self.plots[5]:  # Only bottom plot should show full label
                        plot.getAxis('bottom').setLabel(x_label)
                    
        except Exception as e:
            logger.error(f"Error updating datetime axis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def setup_axis(self, plot, axis_pos, label, units, color):
        """Configure axis with consistent styling"""
        axis = plot.getAxis(axis_pos)
        axis.setPen(pg.mkPen(color))
        axis.setTextPen(color)
        axis.setLabel(text=label, units=units, color=color)

    ######################### for sheath region mode

    def update_region_colors(self):
        """Update region colors based on current mode"""
        try:
            # Define colors
            upstream_color = pg.mkBrush(255, 255, 0, 20)  # Yellow, semi-transparent
            sheath_color = pg.mkBrush(255, 165, 0, 20)    # Orange, semi-transparent
            mo_color = pg.mkBrush(100, 149, 237, 20)      # Light blue, semi-transparent
            
            # Update colors based on mode
            if hasattr(self, 'regions') and self.regions:
                # First region is always upstream
                if len(self.regions) > 0:
                    self.regions[0].setBrush(upstream_color)
                    
                if hasattr(self, 'separate_regions_checkbox') and self.separate_regions_checkbox.isChecked():
                    # Separate mode - main region becomes sheath
                    if hasattr(self, 'sheath_region') and self.sheath_region:
                        self.sheath_region.setBrush(sheath_color)
                        
                    if hasattr(self, 'mo_region') and self.mo_region:
                        self.mo_region.setBrush(mo_color)
                else:
                    # Combined mode - make main region blue color
                    for i, region in enumerate(self.regions):
                        if i > 0:  # Skip upstream region
                            combined_color = pg.mkBrush(180, 180, 220, 20)  # Combined purple
                            region.setBrush(combined_color)
        
        except Exception as e:
            logger.error(f"Error updating region colors: {str(e)}")

    
    def on_separator_moved(self, moved_separator=None):
        """Handle movement of the sheath separator, keeping it within sheath boundaries"""
        try:
            # Find which separator was moved
            if not moved_separator and hasattr(self, 'sheath_separator'):
                moved_separator = self.sheath_separator
                
            if not moved_separator:
                return
                
            # Get current separator position
            separator_pos = moved_separator.value()
            
            # Get current sheath region bounds (using first region)
            if not hasattr(self, 'sheath_regions') or not self.sheath_regions:
                return
                
            sheath_bounds = self.sheath_regions[0].getRegion()
            
            # Constrain separator to stay within sheath region
            if separator_pos < sheath_bounds[0]:
                # Keep separator inside sheath region
                separator_pos = sheath_bounds[0]
            elif separator_pos > sheath_bounds[1]:
                # Keep separator inside sheath region
                separator_pos = sheath_bounds[1]
            
            # Update all separator lines to the same position
            if hasattr(self, 'separator_lines'):
                for sep_line in self.separator_lines:
                    if sep_line != moved_separator:
                        sep_line.blockSignals(True)
                        sep_line.setValue(separator_pos)
                        sep_line.blockSignals(False)
            
            # Update internal datetime representation
            year = self.data_manager.start_date.year
            separator_dt = self.data_manager.doy_to_datetime(year, separator_pos)
            self.region_datetime['separator'] = separator_dt
            
        except Exception as e:
            logger.error(f"Error handling separator movement: {str(e)}")

    
    def update_upstream_regions(self):
        """Update all upstream regions when one changes and update sheath start point"""
        try:
            sender = self.sender()
            if not sender or not hasattr(self, 'upstream_regions'):
                return
                
            region_range = sender.getRegion()
            
            # Update all upstream regions except the sender
            for region in self.upstream_regions:
                if region != sender:
                    region.blockSignals(True)
                    region.setRegion(region_range)
                    region.blockSignals(False)
            
            # Update start of sheath regions to match end of upstream
            if hasattr(self, 'sheath_regions'):
                for sheath_region in self.sheath_regions:
                    current_sheath = sheath_region.getRegion()
                    sheath_region.blockSignals(True)
                    sheath_region.setRegion((region_range[1], current_sheath[1]))
                    sheath_region.blockSignals(False)
            
            # Update stored datetime values
            year = self.data_manager.start_date.year
            self.region_datetime['upstream_start'] = self.data_manager.doy_to_datetime(year, region_range[0])
            self.region_datetime['upstream_end'] = self.data_manager.doy_to_datetime(year, region_range[1])
            self.region_datetime['sheath_start'] = self.region_datetime['upstream_end']
            
        except Exception as e:
            logger.error(f"Error updating upstream regions: {str(e)}")
    
    def update_sheath_regions(self):
        """Update all sheath regions when one changes, updating upstream end and MO start points"""
        try:
            sender = self.sender()
            if not sender or not hasattr(self, 'sheath_regions'):
                return
                
            sheath_range = sender.getRegion()
            
            # Update all sheath regions except the sender
            for region in self.sheath_regions:
                if region != sender:
                    region.blockSignals(True)
                    region.setRegion(sheath_range)
                    region.blockSignals(False)
            
            # Update end of upstream regions to match start of sheath
            if hasattr(self, 'upstream_regions'):
                for upstream_region in self.upstream_regions:
                    current_upstream = upstream_region.getRegion()
                    # Only update if it would create a valid upstream region
                    if sheath_range[0] > current_upstream[0]:
                        upstream_region.blockSignals(True)
                        upstream_region.setRegion((current_upstream[0], sheath_range[0]))
                        upstream_region.blockSignals(False)
                    else:
                        # If it would create an invalid upstream, adjust the sheath start instead
                        valid_start = current_upstream[0] + 0.01
                        sender.blockSignals(True)
                        sender.setRegion((valid_start, sheath_range[1]))
                        sender.blockSignals(False)
                        sheath_range = (valid_start, sheath_range[1])
                        
                        # Update all other sheath regions
                        for other_sheath in self.sheath_regions:
                            if other_sheath != sender:
                                other_sheath.blockSignals(True)
                                other_sheath.setRegion(sheath_range)
                                other_sheath.blockSignals(False)
            
            # Update start of MO regions to match end of sheath
            if hasattr(self, 'mo_regions'):
                for mo_region in self.mo_regions:
                    current_mo = mo_region.getRegion()
                    mo_region.blockSignals(True)
                    mo_region.setRegion((sheath_range[1], current_mo[1]))
                    mo_region.blockSignals(False)
            
            # Update datetime values
            year = self.data_manager.start_date.year
            self.region_datetime['sheath_start'] = self.data_manager.doy_to_datetime(year, sheath_range[0])
            self.region_datetime['sheath_end'] = self.data_manager.doy_to_datetime(year, sheath_range[1])
            self.region_datetime['upstream_end'] = self.region_datetime['sheath_start']
            self.region_datetime['mo_start'] = self.region_datetime['sheath_end']
            
            # Update separator position if needed
            if hasattr(self, 'sheath_separator') and self.sheath_separator:
                separator_pos = self.sheath_separator.value()
                if separator_pos < sheath_range[0] or separator_pos > sheath_range[1]:
                    new_pos = (sheath_range[0] + sheath_range[1]) / 2
                    self.sheath_separator.setValue(new_pos)
                    
                    # Update datetime
                    self.region_datetime['separator'] = self.data_manager.doy_to_datetime(year, new_pos)
                    
        except Exception as e:
            logger.error(f"Error updating sheath regions: {str(e)}")

    def create_sheath_separator(self, position):
        """Create the separator line within the sheath region with label"""
        try:
            # Initialize separator visibility (explicitly set to False initially)
            self.separator_visible = False
            
            # Create a list to store all separator lines
            self.separator_lines = []
            
            # Add separator to all plots
            for i, plot in enumerate(self.plots):
                # Create a separate instance for each plot
                pen = pg.mkPen(color=(100, 0, 150), width=2, style=Qt.DashLine)
                
                # Add label
                if i == 5:
                    sep_line = pg.InfiniteLine(
                        pos=position,
                        angle=90,
                        pen=pen,
                        movable=True,
                        label='drag here',
                        labelOpts={'position': 0.75, 'color': (100, 0, 150), 'fill': (230, 230, 230, 120)}
                    )
                else:
                    sep_line = pg.InfiniteLine(
                        pos=position,
                        angle=90,
                        pen=pen,
                        movable=True
                    )
                
                # Explicitly set visibility to False initially
                sep_line.setVisible(False)
                
                plot.addItem(sep_line)
                
                # Connect movement handler
                sep_line.sigPositionChanged.connect(lambda: self.on_separator_moved(sep_line))
                
                # Store in our list
                self.separator_lines.append(sep_line)
            
            # Set main reference to first separator
            self.sheath_separator = self.separator_lines[0] if self.separator_lines else None
            
        except Exception as e:
            logger.error(f"Error creating sheath separator: {str(e)}")
    
    def update_mo_regions(self):
        """Update all MO regions when one changes, and update sheath end point"""
        try:
            sender = self.sender()
            if not sender or not hasattr(self, 'mo_regions'):
                return
                
            mo_range = sender.getRegion()
            
            # Update all MO regions except the sender
            for region in self.mo_regions:
                if region != sender:
                    region.blockSignals(True)
                    region.setRegion(mo_range)
                    region.blockSignals(False)
            
            # Update end of sheath regions to match start of MO
            if hasattr(self, 'sheath_regions'):
                for sheath_region in self.sheath_regions:
                    current_sheath = sheath_region.getRegion()
                    # Only update if it would create a valid sheath region
                    if mo_range[0] > current_sheath[0]:
                        sheath_region.blockSignals(True)
                        sheath_region.setRegion((current_sheath[0], mo_range[0]))
                        sheath_region.blockSignals(False)
                    else:
                        # If it would create an invalid sheath, adjust the MO start instead
                        valid_start = current_sheath[0] + 0.01
                        sender.blockSignals(True)
                        sender.setRegion((valid_start, mo_range[1]))
                        sender.blockSignals(False)
                        mo_range = (valid_start, mo_range[1])
                        
                        # Update all other MO regions
                        for other_mo in self.mo_regions:
                            if other_mo != sender:
                                other_mo.blockSignals(True)
                                other_mo.setRegion(mo_range)
                                other_mo.blockSignals(False)
            
            # Update datetime values
            year = self.data_manager.start_date.year
            self.region_datetime['mo_start'] = self.data_manager.doy_to_datetime(year, mo_range[0])
            self.region_datetime['mo_end'] = self.data_manager.doy_to_datetime(year, mo_range[1])
            self.region_datetime['sheath_end'] = self.region_datetime['mo_start']
            
        except Exception as e:
            logger.error(f"Error updating MO regions: {str(e)}")

    
    def toggle_separator_visibility(self, checked):
        """Toggle the visibility of the sheath separator"""
        try:
            self.separator_visible = checked
            
            if hasattr(self, 'separator_lines'):
                # If turning on, set to middle of sheath region first
                if checked and hasattr(self, 'sheath_regions') and self.sheath_regions:
                    sheath_bounds = self.sheath_regions[0].getRegion()
                    middle_pos = (sheath_bounds[0] + sheath_bounds[1]) / 2
                    
                    # Update all separator lines
                    for sep_line in self.separator_lines:
                        sep_line.setValue(middle_pos)
                    
                    # Update datetime value
                    year = self.data_manager.start_date.year
                    self.region_datetime['separator'] = self.data_manager.doy_to_datetime(year, middle_pos)
                    
                # Set visibility for all separator lines
                for sep_line in self.separator_lines:
                    sep_line.setVisible(checked)
                
        except Exception as e:
            logger.error(f"Error toggling separator visibility: {str(e)}")


    def throttled_region_update(self):
        """Update all regions simultaneously during dragging - optimized to prevent font scaling issues"""
        try:
            # Get the sender region
            sender = self.sender()
            if not sender:
                return
                
            # Get the region bounds
            region_range = sender.getRegion()
            
            # Determine which type of region collection this belongs to
            if self.analysis_type == "Sheath analysis":
                # For sheath analysis with multiple region types
                if hasattr(self, 'upstream_regions') and sender in self.upstream_regions:
                    self.update_upstream_regions()
                elif hasattr(self, 'sheath_regions') and sender in self.sheath_regions:
                    self.update_sheath_regions()
                elif hasattr(self, 'mo_regions') and sender in self.mo_regions:
                    self.update_mo_regions()
            else:
                # For standard analysis with one region type
                # Update all regions immediately to the same bounds without triggering additional font scaling
                for region in self.regions:
                    if region != sender:
                        region.blockSignals(True)
                        region.setRegion(region_range)
                        region.blockSignals(False)
                
                # Only update labels, not fonts
                self.update_labels(sender)
                    
        except Exception as e:
            logger.error(f"Error in throttled region update: {str(e)}")

    ### TEMPORATL FUNCTION OF SAVING LUNDQUIST TO CSV
    def save_lundquist_results_csv(self, output_dir, event_date, parameters, result):
        """Save Lundquist fit results to CSV"""
        try:
            # Create CSV path
            csv_path = os.path.join(output_dir, "lundquist_results.csv")
            
            # Create data row
            data_row = {
                'date': event_date.strftime('%Y-%m-%d'),
                'satellite': self.data_manager.satellite,
                'theta0': f"{result['optimized_parameters']['theta0']:.2f}",
                'phi0': f"{result['optimized_parameters']['phi0']:.2f}",
                'p0': f"{result['optimized_parameters']['p0']:.3f}",
                'h': f"{result['optimized_parameters']['h']}",
                'b0': f"{result['optimized_parameters']['b0']:.2f}",
                't0': f"{result['optimized_parameters']['t0']:.2f}",
                'r0': f"{result['derived_parameters']['r0']:.3f}",
                'chi_dir': f"{result['derived_parameters']['chi_dir']:.3f}",
                'chi_mag': f"{result['derived_parameters']['chi_mag']:.3f}",
                'flu_axis': f"{result['derived_parameters']['flu_axis']:.2e}",
                'flu_polo': f"{result['derived_parameters']['flu_polo']:.2e}",
                'observer': self.observer_name
            }
            
            # Check if CSV exists already
            import csv
            if os.path.exists(csv_path):
                # Read existing data
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                # Get fieldnames from first row or create new ones
                if rows:
                    fieldnames = reader.fieldnames
                else:
                    fieldnames = list(data_row.keys())
                    
                # Check if we should update an existing row
                updated = False
                for i, row in enumerate(rows):
                    if (row.get('date') == data_row['date'] and 
                        row.get('satellite') == data_row['satellite']):
                        # Update row
                        rows[i] = data_row
                        updated = True
                        break
                        
                # Add row if not updated
                if not updated:
                    rows.append(data_row)
            else:
                # Create new CSV
                fieldnames = list(data_row.keys())
                rows = [data_row]
            
            # Write to CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        except Exception as e:
            logger.error(f"Error saving Lundquist results to CSV: {str(e)}")

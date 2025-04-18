# plot_window.py


from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QGridLayout, 
    QMessageBox, QComboBox, QScrollArea, QSizePolicy,
    QApplication
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from datetime import timedelta
import math
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from calculations import CalculationManager
from fit_window import FitWindow
from output_handler import OutputHandler

logger = logging.getLogger(__name__)

class PlotWindow(QMainWindow):
    def __init__(self, data_manager, observer_name, on_calculate, on_dates_changed):
        super().__init__()
        self.data_manager = data_manager
        self.on_calculate = on_calculate
        self.observer_name = observer_name
        self.on_dates_changed = on_dates_changed
        
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
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """Initialize the UI components with fixed control bar"""
        self.setWindowTitle("Data Analysis")
        
        # Calculate initial window size
        initial_width = min(900, int(self.screen_width * 0.65))
        initial_height = min(1000, int(self.screen_height * 0.85))
        self.resize(initial_width, initial_height)
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        # Create top section widget (info bar + plots)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(3)
        
        # Add info bar
        self.setup_info_bar(top_layout)
        
        # Create scroll area for plots only
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scrollable content widget for plots
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(2)
        
        # Add plots
        self.setup_plots(scroll_layout)
        
        # Set scroll content and add to top section
        scroll_area.setWidget(scroll_content)
        top_layout.addWidget(scroll_area)
        
        # Add top section to main layout
        main_layout.addWidget(top_widget)
        
        # Create and add control panel (fixed at bottom)
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_frame.setMinimumHeight(80)
        self.setup_control_panel(control_frame)
        main_layout.addWidget(control_frame)
        
        # Set minimum sizes
        self.setMinimumWidth(00)
        self.setMinimumHeight(600)
        
    def setup_info_bar(self, parent_layout):
        """Create the information bar at the top with proper null checks"""
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout = QHBoxLayout(info_frame)
        
        # Satellite info with null checks
        satellite_info = f"Satellite: {self.data_manager.satellite or 'Unknown'}"
        if hasattr(self.data_manager, 'distance') and self.data_manager.distance is not None:
            satellite_info += f"  Distance: {self.data_manager.distance:.2f} AU"
    
        # Ensure detector exists in the dictionary
        detector_info = "Detector: Unknown"
        if (hasattr(self.data_manager, 'detector') and 
            self.data_manager.satellite in self.data_manager.detector):
            detector_info = f"Detector: {self.data_manager.detector[self.data_manager.satellite]}"
            
        # Date info with null checks
        date_info = "Period: Unknown"
        if (self.data_manager.start_date is not None and 
            self.data_manager.end_date is not None):
            date_info = (
                f"Period: {self.data_manager.start_date.strftime('%Y/%m/%d')} - "
                f"{self.data_manager.end_date.strftime('%Y/%m/%d')}"
            )
        
        info_label = QLabel(satellite_info)
        detector_label = QLabel(detector_info)
        date_label = QLabel(date_info)
        
        info_layout.addWidget(info_label)
        info_layout.addWidget(detector_label)
        info_layout.addWidget(date_label)
        parent_layout.addWidget(info_frame)

    def setup_plots(self, parent_layout):
        """Set up plots with improved sizing and responsiveness"""
        try:
            plots_frame = QFrame()
            plots_frame.setFrameShape(QFrame.StyledPanel)
            plots_layout = QVBoxLayout(plots_frame)
            plots_layout.setSpacing(1)  # Minimal spacing between plots
            plots_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Calculate initial plot height based on window height
            default_plot_height = 120  # Reduced default height
            
            # Create plot widgets with flexible sizing
            plot_widgets = []
            self.plots = []
            self.view_boxes = []
            
            for i in range(6):
                plot_container = QFrame()
                container_layout = QVBoxLayout(plot_container)
                container_layout.setContentsMargins(1, 1, 1, 1)  # Minimal margins
                
                pw = pg.PlotWidget()
                pw.setBackground('w')
                pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                pw.setMinimumHeight(default_plot_height)
                pw.setMaximumHeight(default_plot_height * 2)  # Allow some expansion
                
                container_layout.addWidget(pw)
                plots_layout.addWidget(plot_container)
                
                # Rest of the plot setup code remains the same...
                plot = pw.plotItem
                plot.showAxis('right')
                view_box = None
                
                if i in [0, 2, 3]:
                    view_box = pg.ViewBox()
                    plot.scene().addItem(view_box)
                    plot.getAxis('right').linkToView(view_box)
                    view_box.setXLink(plot.vb)

                    if i == 2:
                        plot.getAxis('right').setLogMode(True)
                        plot.getAxis('right').setLabel('β')
                    
                    def make_update_function(plot_item, view_box_item):
                        def update():
                            if view_box_item is not None:
                                view_box_item.setGeometry(plot_item.vb.sceneBoundingRect())
                        return update
                    
                    plot.vb.sigResized.connect(make_update_function(plot, view_box))
                else:
                    plot.getAxis('right').setStyle(showValues=False)
                
                # Optimize axis widths
                plot.getAxis('left').setWidth(45)  # Reduced width
                plot.getAxis('right').setWidth(45)  # Reduced width
                
                pw.showGrid(x=False, y=False)
                
                plot_widgets.append(pw)
                self.plots.append(plot)
                self.view_boxes.append(view_box)
                
                if i > 0:
                    pw.setXLink(plot_widgets[0])
            
            parent_layout.addWidget(plots_frame)
            
            self.setup_plot_labels()
            self.setup_selection_regions()
            self.setup_helper_lines()
            self.setup_movable_lines()
    
        except Exception as e:
            logger.error(f"Error setting up plots: {str(e)}")
            raise
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Update plot heights based on new window size
        if hasattr(self, 'plots') and self.plots:
            new_height = max(120, min(200, event.size().height() // 7))
            for plot in self.plots:
                if isinstance(plot, pg.PlotItem):
                    plot.getViewBox().setMaximumHeight(new_height)
                    
    def updateViews(self, plot, view_box):
        if view_box is not None:
            view_box.setGeometry(plot.vb.sceneBoundingRect())
            view_box.linkedViewChanged(plot.vb, view_box.XAxis)
    
    def setup_plot_labels(self):
        """Set up labels for all plots"""
        try:
            label_configs = [
                # B and dB
                {
                    'left': ('B', 'nT'),
                    'right': ('dB', 'nT'),
                    'bottom': None
                },
                # B components
                {
                    'left': ('B', 'nT'),
                    'bottom': None
                },
                # V and Beta
                {
                    'left': ('V', 'km/s'),
                    'right': ('log₂(β)', ''),
                    'bottom': None
                },
                # T and density
                {
                    'left': ('T', 'K'),
                    'right': ('n', 'cm⁻³'),
                    'bottom': None
                },
                # GCR
                {
                    'left': ('GCR', '%'),
                    'bottom': None
                },
                # GCR additional
                {
                    'left': ('GCR', '%'),
                    'bottom': ('DOY', f'{self.data_manager.start_date.year}')
                }
            ]
            
            for plot, config in zip(self.plots, label_configs):
                for side, label in config.items():
                    if label is not None:
                        if isinstance(label, tuple):
                            plot.setLabel(side, label[0], units=label[1])
                        else:
                            plot.setLabel(side, label)
                            
        except Exception as e:
            logger.error(f"Error setting up plot labels: {str(e)}")
    
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
                pos=0,
                angle=0,
                pen=pg.mkPen(color='b', width=1, style=Qt.DashLine)
            )
            if self.view_boxes[2]:  # Add to Beta axis
                self.view_boxes[2].addItem(beta_line)
                
        except Exception as e:
            logger.error(f"Error adding helper lines: {str(e)}")
    
    def setup_movable_lines(self):
        """Set up movable lines for upstream speed calculation"""
        try:
            start_pos = self.data_manager.start_date.timetuple().tm_yday
            
            # Create movable lines with labels
            self.movable_line_start = pg.InfiniteLine(
                pos=start_pos + 0.2,
                angle=90,
                pen=pg.mkPen('purple', width=2),
                movable=True,
                label='upstream',
                labelOpts={'position': 0.2, 'color': 'purple'}
            )
            
            self.movable_line_end = pg.InfiniteLine(
                pos=start_pos + 0.8,
                angle=90,
                pen=pg.mkPen('purple', width=2),
                movable=True,
                label='v',
                labelOpts={'position': 0.2, 'color': 'purple'}
            )
            
            # Create shaded region
            self.upstream_region = pg.LinearRegionItem(
                values=(start_pos + 0.2, start_pos + 0.8),
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
            
        except Exception as e:
            logger.error(f"Error setting up movable lines: {str(e)}")

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


    
    def update_upstream_region(self):
        """Update the shaded region between the movable lines"""
        try:
            start_pos = self.movable_line_start.value()
            end_pos = self.movable_line_end.value()
            self.upstream_region.setRegion((min(start_pos, end_pos), max(start_pos, end_pos)))
        except Exception as e:
            logger.error(f"Error updating upstream region: {str(e)}")
        
        
    def setup_selection_regions(self):
        """Set up basic selection regions synchronized across all panels"""
        try:
            self.regions = []
            start_pos = self.data_manager.start_date.timetuple().tm_yday
            
            # Initialize the labels as class attributes
            self.start_label = None
            self.end_label = None
    
            for plot in self.plots:
                # Create region with initial values
                region = pg.LinearRegionItem(
                    values=(start_pos + 0.5, start_pos + 1.5),
                    brush=pg.mkBrush(100, 100, 255, 50),
                    movable=True
                )
                plot.addItem(region)
                
                # Connect region changes for synchronization
                region.sigRegionChanged.connect(self.update_all_regions)
                self.regions.append(region)
                
        except Exception as e:
            logger.error(f"Error setting up selection regions: {str(e)}")


    
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
            # Continue execution even if label update fails
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        try:
            # Update view box geometries
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
            
            # Update plot heights based on new window size
            if hasattr(self, 'plots') and self.plots:
                new_height = max(120, min(200, event.size().height() // 7))
                for plot in self.plots:
                    if isinstance(plot, pg.PlotItem):
                        plot.getViewBox().setMaximumHeight(new_height)
            
            # Only try to update labels if the regions exist
            if hasattr(self, 'regions') and self.regions and len(self.regions) > 5:
                self.update_labels(self.regions[5])
                
        except Exception as e:
            logger.error(f"Error in resize event: {str(e)}")
    
    def update_all_regions(self):
        """Update all regions to match the changed region"""
        try:
            sender = self.sender()
            if not sender:
                return
                
            region_range = sender.getRegion()
            for region in self.regions:
                if region != sender:
                    region.blockSignals(True)  # Prevent recursion
                    region.setRegion(region_range)
                    region.blockSignals(False)
                    
        except Exception as e:
            logger.error(f"Error updating regions: {str(e)}")
    


    
            
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
        """Plot data with proper alignment of minute and hourly data"""
        try:
            # Define plotting pens
            pen1 = pg.mkPen(color=(0,0,0), width=1)  # black
            pen2 = pg.mkPen(color='b', width=1)      # blue 
            pen4 = pg.mkPen(color='r', width=1)      # red
            pen_green = pg.mkPen(color='darkgreen', width=1)
            pen_gray = pg.mkPen(color='gray', width=1)
            
            # Clear previous plots
            self.clear_plots()
            
            # Get time array for x-axis
            time = data.get('time', np.arange(len(next(iter(data.values())))))
            
            # Helper function to create and position legend
            def add_legend_to_plot(plot):
                legend = plot.addLegend(offset=(-10,10))
                legend.setParentItem(plot.getViewBox())
                return legend
                
            if self.data_manager.satellite in ['Helios1', 'Helios2']:
                self.plot_helios_data(time, data)
            else:
                # Panel 1: B and dB
                legend1 = add_legend_to_plot(self.plots[0])
                if len(data.get('B', [])) > 0:
                    self.plots[0].plot(time, data['B'], pen=pen1, name='B')
                    if self.view_boxes[0] and len(data.get('B_fluct', [])) > 0:
                        plotItem = pg.PlotDataItem(time, data['B_fluct'], pen=pen_gray, name='dB')
                        self.view_boxes[0].addItem(plotItem)
                        # Add secondary axis item to legend
                        legend1.addItem(plotItem, 'dB')
                    self.plots[0].getAxis('left').setLabel('B', 'nT')
                    self.plots[0].getAxis('right').setLabel('dB', 'nT')
                
                # Panel 2: B components
                legend2 = add_legend_to_plot(self.plots[1])
                for component, color, name in zip(['Bx', 'By', 'Bz'], ['r', 'b', 'g'], ['Bx', 'By', 'Bz']):
                    if len(data.get(component, [])) > 0:
                        self.plots[1].plot(time, data[component], pen=pg.mkPen(color), name=name)
                self.plots[1].getAxis('left').setLabel('B', 'nT')
                
                # Panel 3: V and Beta
                legend3 = add_legend_to_plot(self.plots[2])
                if len(data.get('V', [])) > 0:
                    self.plots[2].plot(time, data['V'], pen=pen1, name='V')
                    if self.view_boxes[2] and len(data.get('Beta', [])) > 0:
                        plotItem = pg.PlotDataItem(time, data['Beta'], pen=pen2, name='β')
                        self.view_boxes[2].addItem(plotItem)
                        legend3.addItem(plotItem, 'β')
                    self.plots[2].getAxis('left').setLabel('V', 'km/s')
                    self.plots[2].getAxis('right').setLabel('log₂(β)', '')
                    
                    # Set y-axis range
                    v_data = np.array(data['V'])
                    valid_v = v_data[~np.isnan(v_data)]
                    if len(valid_v) > 0:
                        v_min = np.min(valid_v) - 0.2 * abs(np.min(valid_v))
                        v_max = np.max(valid_v) + 0.2 * np.max(valid_v)
                        self.plots[2].setYRange(v_min, v_max)
                
                # Panel 4: T, T_exp and density
                legend4 = add_legend_to_plot(self.plots[3])
                if len(data.get('T', [])) > 0:
                    # Plot measured temperature
                    self.plots[3].plot(time, data['T'], pen=pen4, name='T')
                    
                    # Calculate and plot T_exp if velocity data is available
                    if len(data.get('V', [])) > 0:
                        v_data = data['V']
                        t_exp = self.temp_func(v_data)
                        self.plots[3].plot(time, t_exp, pen=pen1, name='Texp')
                    
                    # Plot density on secondary axis
                    if self.view_boxes[3] and len(data.get('density', [])) > 0:
                        plotItem = pg.PlotDataItem(time, data['density'], pen=pen2, name='n')
                        self.view_boxes[3].addItem(plotItem)
                        legend4.addItem(plotItem, 'n')
                    
                    self.plots[3].getAxis('left').setLabel('T', 'K')
                    self.plots[3].getAxis('right').setLabel('n', 'cm⁻³')
                
                # Panel 5: GCR
                legend5 = add_legend_to_plot(self.plots[4])
                if len(data.get('GCR', [])) > 0:
                    gcr_data = np.array(data['GCR'])
                    valid_mask = ~np.isnan(gcr_data)
                    if np.any(valid_mask):
                        first_valid = gcr_data[np.where(valid_mask)[0][0]]
                        plot_data = ((gcr_data - first_valid) / first_valid) * 100
                        name = 'GCR' if self.data_manager.satellite not in ['Helios1', 'Helios2'] else 'D5'
                        self.plots[4].plot(
                            time[valid_mask], 
                            plot_data[valid_mask],
                            pen=pen1,
                            symbol='o',
                            symbolSize=4,
                            symbolBrush=pen1.color(),
                            name=name
                        )
                    self.plots[4].getAxis('left').setLabel('GCR', '%')
                
                # Panel 6: Additional GCR channel
                if self.data_manager.satellite in ['Helios1', 'Helios2', 'SolO']:
                    legend6 = add_legend_to_plot(self.plots[5])
                    gcr_additional = data.get('GCR_additional', [])
                    if len(gcr_additional) > 0:
                        valid_mask = ~np.isnan(gcr_additional)
                        if np.any(valid_mask):
                            first_valid = gcr_additional[np.where(valid_mask)[0][0]]
                            plot_data = ((gcr_additional - first_valid) / first_valid) * 100
                            
                            # Set name based on satellite type
                            name = '2C_L' if self.data_manager.satellite == 'SolO' else 'CE'
                            
                            self.plots[5].plot(
                                time[valid_mask],
                                plot_data[valid_mask],
                                pen=pen1,
                                symbol='o',
                                symbolSize=4,
                                symbolBrush=pen1.color(),
                                name=name
                            )
                        self.plots[5].getAxis('left').setLabel('GCR', '%')
    
            # Set X axis labels
            hours_per_day = 24
            ticks = []
            min_doy = np.min(time)
            max_doy = np.max(time)
            
            # Create ticks every 12 hours
            for doy in np.arange(np.floor(min_doy), np.ceil(max_doy), 0.5):
                tick_time = doy
                tick_str = f"{doy:.1f}"
                ticks.append((tick_time, tick_str))
            
            # Set ticks for all plots
            for plot in self.plots:
                plot.getAxis('bottom').setTicks([ticks])
                
            # Update view boxes
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
                    
        except Exception as e:
            logger.error(f"Error plotting data: {str(e)}")
            raise

    def temp_func(self, Vf):
        """Calculate expected temperature from velocity with proper NaN handling"""
        T = []
        try:
            for Vp in Vf:
                if np.isnan(Vp):
                    T.append(np.nan)
                elif Vp < 500:
                    Texp = ((0.031 * Vp - 5.100) ** 2) * 10**3
                    T.append(Texp)
                else:
                    Texp = (0.51 * Vp - 142) * 10**3
                    T.append(Texp)
        except Exception as e:
            logger.error(f"Error calculating T_exp: {str(e)}")
            T = [np.nan] * len(Vf)
        return T

    def plot_helios_data(self, time, data):
        """Plot data for Helios satellites with improved gap and legend handling"""
        pen1 = pg.mkPen(color=(0,0,0), width=1)      # black
        pen2 = pg.mkPen(color='b', width=1)          # blue 
        pen4 = pg.mkPen(color='r', width=1)          # red
        pen_green = pg.mkPen(color='darkgreen', width=1)
        pen_gray = pg.mkPen(color='gray', width=1)
    
        try:
            def plot_with_gaps(plot_or_viewbox, x, y, pen, name=None, symbol_size=3, max_gap=0.1):
                """Plot data with proper gap handling and legend support"""
                x = np.array(x)
                y = np.array(y)
                mask = ~np.isnan(y)
                x_valid = x[mask]
                y_valid = y[mask]
                
                if len(x_valid) == 0:
                    return None
                    
                gap_indices = np.where(np.diff(x_valid) > max_gap)[0]
                
                if len(gap_indices) == 0:
                    plot_item = pg.PlotDataItem(
                        x_valid, 
                        y_valid,
                        pen=pen,  
                        symbol='o',
                        symbolSize=symbol_size,
                        symbolBrush=pen.color(),
                        symbolPen=None,
                        name=name
                    )
                    if isinstance(plot_or_viewbox, pg.PlotItem):
                        plot_or_viewbox.addItem(plot_item)
                    else:  # ViewBox
                        plot_or_viewbox.addItem(plot_item)
                    return plot_item
    
                # For data with gaps, only return the first plot item for legend
                first_item = None
                start_idx = 0
                for gap_idx in gap_indices:
                    end_idx = gap_idx + 1
                    if start_idx < end_idx:
                        plot_item = pg.PlotDataItem(
                            x_valid[start_idx:end_idx], 
                            y_valid[start_idx:end_idx],
                            pen=pen,  
                            symbol='o',
                            symbolSize=symbol_size,
                            symbolBrush=pen.color(),
                            symbolPen=None,
                            name=name if first_item is None else None
                        )
                        if isinstance(plot_or_viewbox, pg.PlotItem):
                            plot_or_viewbox.addItem(plot_item)
                        else:
                            plot_or_viewbox.addItem(plot_item)
                        if first_item is None:
                            first_item = plot_item
                    start_idx = end_idx
    
                # Plot final segment
                if start_idx < len(x_valid):
                    plot_item = pg.PlotDataItem(
                        x_valid[start_idx:], 
                        y_valid[start_idx:],
                        pen=pen,
                        symbol='o',
                        symbolSize=symbol_size,
                        symbolBrush=pen.color(),
                        symbolPen=None,
                        name=name if first_item is None else None
                    )
                    if isinstance(plot_or_viewbox, pg.PlotItem):
                        plot_or_viewbox.addItem(plot_item)
                    else:
                        plot_or_viewbox.addItem(plot_item)
                    if first_item is None:
                        first_item = plot_item
    
                return first_item
    
            # Helper function to create and position legend
            def add_legend_to_plot(plot):
                legend = plot.addLegend(offset=(-10,10))
                legend.setParentItem(plot.getViewBox())
                return legend
    
            # Panel 1: B and dB
            legend1 = add_legend_to_plot(self.plots[0])
            if len(data.get('B', [])) > 0:
                main_plot = plot_with_gaps(self.plots[0], time, data['B'], pen1, name='B')
                if self.view_boxes[0] and len(data.get('B_fluct', [])) > 0:
                    fluct_plot = plot_with_gaps(self.view_boxes[0], time, data['B_fluct'], 
                                              pen_gray, name='dB')
                    if fluct_plot:
                        legend1.addItem(fluct_plot, 'dB')
    
            # Panel 2: B components
            legend2 = add_legend_to_plot(self.plots[1])
            for component, color, name in zip(['Bx', 'By', 'Bz'], ['r', 'b', 'g'], ['Bx', 'By', 'Bz']):
                if len(data.get(component, [])) > 0:
                    plot_with_gaps(self.plots[1], time, data[component], 
                                 pg.mkPen(color), name=name)
    
            # Panel 3: V and Beta
            legend3 = add_legend_to_plot(self.plots[2])
            if len(data.get('V', [])) > 0:
                v_plot = plot_with_gaps(self.plots[2], time, data['V'], pen1, name='V')
                if self.view_boxes[2] and len(data.get('Beta', [])) > 0:
                    beta_plot = plot_with_gaps(self.view_boxes[2], time, data['Beta'], 
                                             pen2, name='β')
                    if beta_plot:
                        legend3.addItem(beta_plot, 'β')
    
            # Panel 4: T, T_exp, and density
            legend4 = add_legend_to_plot(self.plots[3])
            if len(data.get('T', [])) > 0:
                # Plot measured temperature
                t_plot = plot_with_gaps(self.plots[3], time, data['T'], pen4, name='T')
                
                # Calculate and plot T_exp
                if len(data.get('V', [])) > 0:
                    v_data = data['V']
                    t_exp = self.temp_func(v_data)
                    plot_with_gaps(self.plots[3], time, t_exp, pen1, name='Texp')
                
                # Plot density
                if self.view_boxes[3] and len(data.get('density', [])) > 0:
                    n_plot = plot_with_gaps(self.view_boxes[3], time, data['density'], 
                                          pen2, name='n')
                    if n_plot:
                        legend4.addItem(n_plot, 'n')
    
            # Panel 5: GCR D5
            legend5 = add_legend_to_plot(self.plots[4])
            if len(data.get('GCR', [])) > 0:
                gcr_data = np.array(data['GCR'])
                valid_mask = ~np.isnan(gcr_data)
                if np.any(valid_mask):
                    first_valid = gcr_data[np.where(valid_mask)[0][0]]
                    plot_data = ((gcr_data - first_valid) / first_valid) * 100
                    plot_with_gaps(self.plots[4], time, plot_data, pen1, 
                                 name='GCR D5', symbol_size=4)
    
            # Panel 6: GCR CE
            legend6 = add_legend_to_plot(self.plots[5])
            if len(data.get('GCR_additional', [])) > 0:
                gcr_add = np.array(data['GCR_additional'])
                valid_mask = ~np.isnan(gcr_add)
                if np.any(valid_mask):
                    first_valid = gcr_add[np.where(valid_mask)[0][0]]
                    plot_data = ((gcr_add - first_valid) / first_valid) * 100
                    plot_with_gaps(self.plots[5], time, plot_data, pen1, 
                                 name='GCR CE', symbol_size=4)
    
        except Exception as e:
            logger.error(f"Error plotting Helios data: {str(e)}")
            raise

    
            
                
    def setup_control_panel(self, parent_frame):
        """Set up the control panel with original navigation buttons"""
        control_layout = QHBoxLayout(parent_frame)
        control_layout.setContentsMargins(15, 12, 15, 12)
        
        # Navigation buttons
        nav_buttons = [
            ("◀ 5 days", lambda: self.adjust_dates(-5)),
            ("◀ 2 days", lambda: self.adjust_dates(-2)),
            ("2 days ▶", lambda: self.adjust_dates(2)),
            ("5 days ▶", lambda: self.adjust_dates(5))
        ]
        
        for text, callback in nav_buttons:
            btn = QPushButton(text)
            btn.setFixedWidth(100)
            btn.setFixedHeight(45)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:checked {
                background-color: #1565C0;
            }
            """)
            control_layout.addWidget(btn)
            
        # Add stretch to push the right side elements to the right
        control_layout.addStretch()
        
        # Fit type selection
        self.fit_type = QComboBox()
        self.fit_type.addItems(["Test", "Inner", "External", "Optimal"])
        self.fit_type.setFixedWidth(150)
        self.fit_type.setFixedHeight(45)
        self.fit_type.setStyleSheet("""
            QComboBox {
                border: 1px solid #BDBDBD;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
                font-size: 14px;
            }
            QComboBox:hover {
                border: 1px solid #757575;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid #757575;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                font-size: 14px;
                border: 1px solid #BDBDBD;
                background-color: white;
                selection-background-color: #E0E0E0;
                selection-color: black;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 35px;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #EEEEEE;
                color: black;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #E0E0E0;
                color: black;
            }
        """)
        control_layout.addWidget(self.fit_type)
        
        # Calculate button
        calc_btn = QPushButton("Calculate")
        calc_btn.setFixedWidth(150)
        calc_btn.setFixedHeight(45)
        calc_btn.clicked.connect(self.on_calculate_clicked)
        calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a365d;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
            background-color: #2c4c8c;  
            }
            QPushButton:pressed {
            background-color: #152a4a;  
            }
        """)
        control_layout.addWidget(calc_btn)
        
        
    def load_data(self):
        """Load and display the data with specific warnings"""
        try:
            logger.info("Loading data for plotting")
            print("About to load data...")
            data = self.data_manager.load_data()
            
            print("Data loaded, type:", type(data))
            if isinstance(data, dict):
                print("Data keys:", data.keys())
            
            if not data:
                logger.warning("No data loaded")
                return
                
            print("About to plot data...")
            self.plot_data(data)
            print("Data plotted successfully")
    
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            

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


    def on_calculate_clicked(self):
        """Handle calculate button click event with direct saving when no GCR data"""
        try:
            # Validate selections
            if not self.regions:
                QMessageBox.warning(self, "Warning", "Please define the ICME borders")
                return
    
            if not (self.movable_line_start and self.movable_line_end):
                QMessageBox.warning(self, "Warning", "Please set the upstream window borders (purple lines)")
                return
    
            # Get region bounds and line positions
            region = self.regions[0].getRegion()
            upstream_window = (self.movable_line_start.value(), self.movable_line_end.value())
    
            # Get current fit type and prepare output directory
            current_fit_type = self.fit_type.currentText()
            results_dir = self.data_manager.create_output_directory(current_fit_type)
            
            # Ensure output directories exist
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
    
            # Create calculator instance
            calculator = CalculationManager(self.data_manager)
            
            # Perform calculations
            calc_results = calculator.perform_calculations(
                region[0], region[1],  # ICME window
                upstream_window[0], upstream_window[1]  # Upstream window
            )
    
            # Check for GCR data availability
            if not calc_results.get('has_gcr_data', False):
                # Show warning
                QMessageBox.warning(self, "Warning", 
                    "No GCR data available. Only ICME parameters will be saved.")
    
                # Prepare output info
                output_info = {
                    'script_directory': os.path.dirname(os.path.abspath(__file__)),
                    'results_directory': results_dir,
                    'day': self.data_manager.start_date.strftime('%Y/%m/%d'),
                    'fit': current_fit_type.lower(),
                    'observer_name': self.observer_name,
                    'data_manager': self.data_manager 
                }
    
                # Save the plot window
                self.save_plot_window(results_dir)
    
                # Create output handler and save results directly
                output_handler = OutputHandler(results_dir, output_info['script_directory'])
                
                # Save parameters and update CSV
                output_handler.save_parameters(output_info, calc_results)
                output_handler.update_results_csv(
                    sat=self.data_manager.satellite,
                    detector=self.data_manager.detector,
                    observer=self.observer_name,
                    calc_results=calc_results,
                    day=output_info['day'],
                    fit_type=current_fit_type,
                    fit_categories=['no GCR data']
                )
                
                QMessageBox.information(self, "Success", 
                    "ICME parameters have been saved successfully.")
                return
    
            # If we have GCR data, proceed with fit window as normal
            output_info = {
                'script_directory': os.path.dirname(os.path.abspath(__file__)),
                'results_directory': results_dir,
                'day': self.data_manager.start_date.strftime('%Y/%m/%d'),
                'fit': current_fit_type.lower(),
                'observer_name': self.observer_name,
                'data_manager': self.data_manager 
            }
    
            # Save the plot window
            self.save_plot_window(results_dir)
    
            # Show fit window with calculation results
            self.fit_window = FitWindow(
                sat=self.data_manager.satellite,
                detector=self.data_manager.detector,
                observer=self.observer_name,
                calc_results=calc_results,
                output_info=output_info
            )
            self.fit_window.show()
    
        except Exception as e:
            logger.error(f"Error during calculation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")
    
    def save_plot_window(self, results_dir):
        """Save the current plot window state with adjusted figure size"""
        try:
            # Reduce figure size (original was 14, 16)
            fig = plt.figure(figsize=(10, 12))
            
            # Create GridSpec with smaller spacing between subplots
            gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.15)
            
            # Get the boundaries
            region_bounds = self.regions[0].getRegion()
            upstream_start = self.movable_line_start.value()
            upstream_end = self.movable_line_end.value()
            
            # Get data
            data = self.data_manager.load_data()
            if not data:
                raise ValueError("No data available for plotting")
    
            # Get time data and validate
            time = data.get('time', [])
            if len(time) == 0:
                time = np.arange(len(next(iter(data.values()))))
    
            # Set time limits for x-axis
            time_min = np.min(time) if len(time) > 0 else region_bounds[0] - 1
            time_max = np.max(time) if len(time) > 0 else region_bounds[1] + 1
    
            # Loop through plots
            for i in range(6):
                # Create subplot using GridSpec
                ax = fig.add_subplot(gs[i])
                ax2 = None
                
            # Add twin axis for specific panels
                if i in [0, 1, 2, 3]:
                    ax2 = ax.twinx()
    
                # Plot data based on panel
                if i == 0:  # B and dB
                    if 'B' in data and len(data['B']) > 0:
                        b_data = np.array(data['B'])
                        valid_mask = ~np.isnan(b_data)
                        if np.any(valid_mask):
                            ax.plot(time[valid_mask], b_data[valid_mask], 'k-', label='B')
                            ax.set_ylabel('B [nT]')
                    
                    if 'B_fluct' in data and len(data['B_fluct']) > 0 and ax2:
                        b_fluct = np.array(data['B_fluct'])
                        valid_mask = ~np.isnan(b_fluct)
                        if np.any(valid_mask):
                            ax2.plot(time[valid_mask], b_fluct[valid_mask], color='gray', label='dB')
                            ax2.set_ylabel('dB [nT]')
    
                elif i == 1:  # B components
                    for comp, color, label in zip(['Bx', 'By', 'Bz'], ['r', 'b', 'g'], ['Bx', 'By', 'Bz']):
                        if comp in data and len(data[comp]) > 0:
                            comp_data = np.array(data[comp])
                            valid_mask = ~np.isnan(comp_data)
                            if np.any(valid_mask):
                                ax.plot(time[valid_mask], comp_data[valid_mask], color=color, label=label)
                    ax.set_ylabel('B [nT]')
    
                elif i == 2:  # V and Beta
                    if 'V' in data and len(data['V']) > 0:
                        v_data = np.array(data['V'])
                        valid_mask = ~np.isnan(v_data)
                        if np.any(valid_mask):
                            ax.plot(time[valid_mask], v_data[valid_mask], 'k-', label='V')
                            ax.set_ylabel('V [km/s]')
                    
                    if 'Beta' in data and len(data['Beta']) > 0 and ax2:
                        beta_data = np.array(data['Beta'])
                        valid_mask = ~np.isnan(beta_data)
                        if np.any(valid_mask):
                            ax2.plot(time[valid_mask], beta_data[valid_mask], 'b-', label='β')
                            ax2.set_ylabel('log₂(β)')
    
                elif i == 3:  # T and density
                    if 'T' in data and len(data['T']) > 0:
                        t_data = np.array(data['T'])
                        valid_mask = ~np.isnan(t_data)
                        if np.any(valid_mask):
                            ax.plot(time[valid_mask], t_data[valid_mask], 'r-', label='T')
                            ax.set_ylabel('T [K]')
                    
                    if 'density' in data and len(data['density']) > 0 and ax2:
                        n_data = np.array(data['density'])
                        valid_mask = ~np.isnan(n_data)
                        if np.any(valid_mask):
                            ax2.plot(time[valid_mask], n_data[valid_mask], 'b-', label='n')
                            ax2.set_ylabel('n [cm⁻³]')
    
                elif i in [4, 5]:  # GCR plots
                    gcr_key = 'GCR' if i == 4 else 'GCR_additional'
                    if gcr_key in data and len(data[gcr_key]) > 0:
                        gcr_data = np.array(data[gcr_key])
                        valid_mask = ~np.isnan(gcr_data)
                        if np.any(valid_mask):
                            first_valid = gcr_data[np.where(valid_mask)[0][0]]
                            plot_data = ((gcr_data - first_valid) / first_valid) * 100
                            ax.plot(time[valid_mask], plot_data[valid_mask], 'ko-', 
                                   markersize=4, label=gcr_key)
                    ax.set_ylabel('GCR [%]')
    
                # Add ICME boundaries shading
                ax.axvspan(region_bounds[0], region_bounds[1], alpha=0.2, color='blue', label='ICME')
                
                # Add upstream window shading to speed panel
                if i == 2:
                    ax.axvspan(upstream_start, upstream_end, alpha=0.2, color='purple', label='Upstream')
    
                # Set x-axis limits
                ax.set_xlim(time_min, time_max)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
    
                # Make fonts smaller
                ax.tick_params(labelsize=8)
                if ax2:
                    ax2.tick_params(labelsize=8)
                ax.yaxis.label.set_size(9)
                if ax2:
                    ax2.yaxis.label.set_size(9)
                
                # Smaller legend
                handles1, labels1 = ax.get_legend_handles_labels()
                if ax2:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles = handles1 + handles2
                    labels = labels1 + labels2
                else:
                    handles = handles1
                    labels = labels1
                
                if handles:
                    ax.legend(handles, labels, loc='upper right', fontsize=7, 
                             bbox_to_anchor=(1.0, 1.0))
    
                # Set x-label for bottom plot only
                if i == 5:
                    ax.set_xlabel(f'DOY [{self.data_manager.start_date.year}]', fontsize=9)
                
                # Set title for top plot
                if i == 0:
                    ax.set_title(f"{self.data_manager.satellite} {self.data_manager.start_date.strftime('%Y/%m/%d')}", 
                                fontsize=10, pad=3)
    
            # Adjust layout with tighter spacing
            plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
            
            # Save figure with reduced DPI
            plt.savefig(os.path.join(results_dir, 'plot_window.png'), 
                       dpi=200, bbox_inches='tight')
            plt.close(fig)
    
        except Exception as e:
            logger.error(f"Error saving plot window: {str(e)}")
            raise


    def adjust_dates(self, days):
        """Handle date navigation"""
        try:
            new_start = self.data_manager.start_date + timedelta(days=days)
            new_end = self.data_manager.end_date + timedelta(days=days)
            
            # Clear plots before changing dates to avoid artifacts
            self.clear_plots()
            
            # Update dates and reload data
            self.on_dates_changed(new_start, new_end)
            
            # Ensure view boxes are properly updated
            for plot, view_box in zip(self.plots, self.view_boxes):
                if view_box is not None:
                    self.updateViews(plot, view_box)
                
        except Exception as e:
            logger.error(f"Error adjusting dates: {str(e)}")
            QMessageBox.warning(self, "Warning", "Failed to adjust dates")
        
    def clear_plots(self):
        """Clear all plots including view boxes"""
        try:
            self.regions = []
            
            for plot in self.plots:
                plot.clear()
                
            for view_box in self.view_boxes:
                if view_box is not None:
                    for item in view_box.addedItems[:]:
                        view_box.removeItem(item)
                        
            self.setup_helper_lines()
            self.setup_selection_regions()
            self.setup_movable_lines()
        
        except Exception as e:
            logger.error(f"Error clearing plots: {str(e)}")

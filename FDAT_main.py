# main.py


# Set High DPI scaling before importing PyQt
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
import qdarktheme

# Set High DPI attributes BEFORE QApplication is created
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

import os

os.environ['SPACEPY_LEAPSECS_WARN'] = 'False'
#os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.xcb=false'

try:
    import spacepy.toolbox
    spacepy.toolbox.update(leapsecs=True)
except Exception as e:
    print(f"Could not update leap seconds: {e}")

# to handle matplotlib configs
from matplotlib_setup import configure_matplotlib
configure_matplotlib()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox,
    QFrame, QGridLayout, QMessageBox, QSizePolicy, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QFont
from datetime import datetime, timedelta
import logging
import math
import json




# Configure logging 
logging.basicConfig(
    level=logging.ERROR,  # Change from DEBUG to ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('FDAT.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Silence matplotlib warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
logging.getLogger('matplotlib.legend').setLevel(logging.ERROR)

# Import custom modules
from plot_window import PlotWindow
from cdf_data_manager import CDFDataManager
from settings_manager import SettingsManager
from utils import WindowManager
from icon import apply_app_icon


class StartWindow(QMainWindow):
    def __init__(self, callback, window_manager=None):
        super().__init__()
        # Store the full callback
        self.callback = callback
        self.selected_satellite = None
        self.settings_manager = SettingsManager()
        self.window_manager = window_manager
        self.on_dates_changed = None
        
        # Initialize radio button references
        self.forbmod_radio = None
        self.insitu_radio = None
        self.sheath_radio = None
        
        # Load satellite mappings directly without creating a full data manager
        self.satellite_dates = self.load_satellite_dates()
        
        self.setup_ui()
        
        # Apply saved window geometry if available
        if self.window_manager:
            self.window_manager.apply_window_geometry(self, 'start_window')
    
    def load_satellite_dates(self):
        """Load satellite date ranges directly from JSON file"""
        satellite_dates = {}
        try:
            mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'satellite_mappings.json')
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
                    if 'satellites' in mappings:
                        for sat_name, sat_data in mappings['satellites'].items():
                            if 'date_range' in sat_data:
                                satellite_dates[sat_name] = sat_data['date_range']
        except Exception as e:
            print(f"Error loading satellite dates: {e}")
        
        return satellite_dates
            


    def show_start_window(self):
        self.start_window = StartWindow(
            callback=lambda sat, start, end, obs, analysis: self.on_start_window_complete(
                sat, start, end, obs, analysis
            )
        )

        # Apply geometry then show
        self.window_manager.apply_window_geometry(self.window, 'start_window')
        self.window.show()


    def closeEvent(self, event):
        """Override close event to save window geometry"""
        if self.window_manager:
            self.window_manager.save_window_geometry(self, 'start_window')
        super().closeEvent(event)
        
    def setup_ui(self):
        self.setWindowTitle("ForbMod Analysis Tool")
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Date selection
        date_frame = self.create_date_section()
        layout.addWidget(date_frame)
        
        # Satellite selection
        satellite_frame = self.create_satellite_section()
        layout.addWidget(satellite_frame)
        
        # Analyze button with direct font setting
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setFont(self.window_manager.get_sized_font('header'))
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)
        layout.addWidget(self.analyze_btn)
        
        self.apply_styles()
    
    def save_analysis_type(self, button):
        """Save analysis type when radio button changes"""
        analysis_types = {
            self.forbmod_radio: "ForbMod",
            self.insitu_radio: "In-situ analysis",
            self.sheath_radio: "Sheath analysis"
        }
        
        # Get the analysis type, defaulting to ForbMod if not found
        analysis_type = analysis_types.get(button, "ForbMod")
        
        self.settings_manager.set_analysis_type(analysis_type)
    


    def create_header(self):
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(header)
        
        # Create title with title font directly
        title = QLabel("Forbush Decrease Analysis Tool")
        title.setFont(self.window_manager.get_sized_font('title'))
        title.setAlignment(Qt.AlignCenter)
        
        # Create description with normal font
        description = QLabel("Analyze ICME and Forbush Decrease events")
        description.setFont(self.window_manager.get_sized_font('normal'))
        description.setAlignment(Qt.AlignCenter)
    
        # Combined observer and analysis type row
        input_layout = QHBoxLayout()
        
        # Observer part (left side)
        observer_layout = QHBoxLayout()
        observer_label = QLabel("Observer:")
        observer_label.setFont(self.window_manager.get_sized_font('normal'))
        observer_label.setFixedWidth(100)
        
        self.observer_input = QLineEdit()
        self.observer_input.setText(self.settings_manager.get_observer_name())
        self.observer_input.textChanged.connect(self.save_observer_name)
        self.observer_input.setFont(self.window_manager.get_sized_font('normal'))
        
        observer_layout.addWidget(observer_label)
        observer_layout.addWidget(self.observer_input)
        input_layout.addLayout(observer_layout, stretch=2)
        input_layout.addSpacing(20)
        
        # Analysis type part - radio buttons
        analysis_layout = QHBoxLayout()
        analysis_label = QLabel("Analysis type:")
        analysis_label.setFont(self.window_manager.get_sized_font('normal'))
        
        self.forbmod_radio = QRadioButton("ForbMod")
        self.insitu_radio = QRadioButton("In-situ")
        self.sheath_radio = QRadioButton("Sheath")  
        self.lundquist_radio = QRadioButton("Lundquist fit") 

           
    


        
        self.forbmod_radio.setFont(self.window_manager.get_sized_font('normal'))
        self.insitu_radio.setFont(self.window_manager.get_sized_font('normal'))
        self.sheath_radio.setFont(self.window_manager.get_sized_font('normal'))
        self.lundquist_radio.setFont(self.window_manager.get_sized_font('normal'))
        
        analysis_layout.addWidget(analysis_label)
        analysis_layout.addWidget(self.forbmod_radio)
        analysis_layout.addWidget(self.insitu_radio)
        analysis_layout.addWidget(self.sheath_radio) 
        analysis_layout.addWidget(self.lundquist_radio)  
        input_layout.addLayout(analysis_layout, stretch=1)  
        
        # Group radio buttons
        button_group = QButtonGroup(self)
        button_group.addButton(self.forbmod_radio)
        button_group.addButton(self.insitu_radio)
        button_group.addButton(self.sheath_radio)  
        button_group.addButton(self.lundquist_radio) 
        
        # Set default based on saved preference
        saved_type = self.settings_manager.get_analysis_type()
        if saved_type == "In-situ analysis":
            self.insitu_radio.setChecked(True)
        elif saved_type == "Sheath analysis":  
            self.sheath_radio.setChecked(True)
        elif saved_type == "Lundquist fit":
            self.lundquist_radio.setChecked(True)
        else:
            self.forbmod_radio.setChecked(True)
        
        # Connect radio button changes
        button_group.buttonClicked.connect(self.save_analysis_type)
        
        # Add all components to main layout
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(input_layout)
        layout.addSpacing(10)
        
        return header
    
    def save_sheath_preference(self, state):
        """Save sheath analysis preference when checkbox changes"""
        if hasattr(self.settings_manager, 'set_sheath_analysis') and callable(self.settings_manager.set_sheath_analysis):
            self.settings_manager.set_sheath_analysis(state == Qt.Checked)
        else:
            # If the method doesn't exist in settings_manager, we need to add it or create a fallback
            print("Note: settings_manager doesn't have set_sheath_analysis method")
    

    def create_satellite_section(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Create label with header font directly
        label = QLabel("Select Satellite:")
        label.setFont(self.window_manager.get_sized_font('header'))
        layout.addWidget(label)
        
        grid = QGridLayout()
        
        # Define the specific order of satellites
        ordered_satellites = [
            # Row 1
            "WIND", "ACE", "SolO",
            # Row 2
            "OMNI", "neutron monitors", "MAVEN",
            # Row 3
            "Ulysses", "Helios1", "Helios2"
        ]
        
        # Default date ranges as fallback
        default_dates = {
            "OMNI": "Jan 1998 - Dec 2024",
            "ACE": "Sep 1997 - Dec 2022",
            "WIND": "Nov 1994 - Sep 2024",
            "SolO": "Apr 2020 - Jul 2024",
            "Helios1": "Dec 1974 - Jun 1981",
            "Helios2": "Jan 1976 - Mar 1980",
            "Ulysses": "1990 - 2009",
            "neutron monitors": "Jan 1998 - Dec 2024",
            "MAVEN": "Nov 2014 - Dec 2024"
        }
        
        self.satellite_buttons = {}
        for i, sat in enumerate(ordered_satellites):
            # Use date from JSON if available, otherwise use default
            years = self.satellite_dates.get(sat, default_dates.get(sat, ""))
            
            btn = QPushButton(f"{sat}\n{years}")
            # Set font directly
            btn.setFont(self.window_manager.get_sized_font('normal'))
            
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=sat: self.select_satellite(s))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(60)
            
            # Calculate row and column based on the predefined order
            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)
            
            self.satellite_buttons[sat] = btn
            
        layout.addLayout(grid)
        return frame
    
    def save_observer_name(self):
        """Save observer name when changed"""
        self.settings_manager.set_observer_name(self.observer_input.text())

    def save_dates(self):
        """Save dates when manually entered"""
        try:
            start_date = datetime.strptime(self.start_date_input.text(), "%Y/%m/%d")
            end_date = datetime.strptime(self.end_date_input.text(), "%Y/%m/%d")
            
            # Only save if both dates are valid and start is before end
            if start_date < end_date:
                self.settings_manager.set_last_dates(start_date, end_date)
        except ValueError:
            pass
        
    def create_date_section(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QGridLayout(frame)
        
        # Create labels with normal font directly
        start_label = QLabel("Start Date (yyyy/mm/dd):")
        start_label.setFont(self.window_manager.get_sized_font('normal'))
        
        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("YYYY/MM/DD")
        self.start_date_input.textChanged.connect(lambda: self.save_dates())
        self.start_date_input.setFont(self.window_manager.get_sized_font('normal'))
        
        end_label = QLabel("End Date (yyyy/mm/dd):")
        end_label.setFont(self.window_manager.get_sized_font('normal'))
        
        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("YYYY/MM/DD")
        self.end_date_input.textChanged.connect(lambda: self.save_dates())
        self.end_date_input.setFont(self.window_manager.get_sized_font('normal'))
        
        # Get saved dates
        saved_start_date, saved_end_date = self.settings_manager.get_last_dates()
        
        if saved_start_date and saved_end_date:
            self.start_date_input.setText(saved_start_date.strftime("%Y/%m/%d"))
            self.end_date_input.setText(saved_end_date.strftime("%Y/%m/%d"))
        else:
            # Default dates if no saved dates exist
            self.start_date_input.setText("1997/10/09")
            self.end_date_input.setText("1997/10/14")
        
        layout.addWidget(start_label, 0, 0)
        layout.addWidget(self.start_date_input, 0, 1)
        layout.addWidget(end_label, 1, 0)
        layout.addWidget(self.end_date_input, 1, 1)
        
        return frame

        
    def apply_styles(self):
        """Apply stylesheets with proper scaling"""
        # Get scaling factors
        dpi_factor = 1.0
        font_scale = 1.0
        if self.window_manager:
            dpi_factor = self.window_manager.settings.get('dpi_factor', 1.0)
            font_scale = self.window_manager.settings.get('font_scale', 1.0)
        
        base_stylesheet = """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QFrame {
                background-color: white;
                border-radius: 5px;
                margin: 5px;
                padding: 10px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:checked {
                background-color: #1565C0;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: %dpx;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: %dpx;
            }
            QLabel {
                color: #333;
                font-size: %dpx;
            }
            QRadioButton {
                font-size: %dpx;
            }
        """ % (
            int(16 * font_scale),  # Font size for QLineEdit
            int(16 * font_scale),  # Font size for QComboBox
            int(16 * font_scale),  # Font size for QLabel
            int(16 * font_scale)   # Font size for QRadioButton
        )
        
        # Use window manager to scale the stylesheet if it's available
        if self.window_manager:
            scaled_stylesheet = self.window_manager.get_scaled_style_sheet(base_stylesheet)
            self.setStyleSheet(scaled_stylesheet)
        else:
            self.setStyleSheet(base_stylesheet)
        
        # Apply font scaling to all widgets
        if self.window_manager:
            self.window_manager.apply_font_scaling(self)
        
    def select_satellite(self, satellite):
        for sat, btn in self.satellite_buttons.items():
            btn.setChecked(sat == satellite)
        self.selected_satellite = satellite
        
    def adjust_dates(self, days):
        try:
            start_date = datetime.strptime(self.start_date_input.text(), "%Y/%m/%d")
            end_date = datetime.strptime(self.end_date_input.text(), "%Y/%m/%d")
            
            start_date += timedelta(days=days)
            end_date += timedelta(days=days)
            
            self.start_date_input.setText(start_date.strftime("%Y/%m/%d"))
            self.end_date_input.setText(end_date.strftime("%Y/%m/%d"))
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid dates first.")
            
    
    def on_analyze_clicked(self):
        try:
            if not self.selected_satellite:
                QMessageBox.warning(self, "Warning", "Please select a satellite first.")
                return
                    
            start_date = datetime.strptime(self.start_date_input.text(), "%Y/%m/%d")
            end_date = datetime.strptime(self.end_date_input.text(), "%Y/%m/%d")
            
            if end_date <= start_date:
                QMessageBox.warning(self, "Warning", "End date must be after start date.")
                return
            
            # determine analysis type from radio buttons
            if self.insitu_radio.isChecked():
                analysis_type = "In-situ analysis"
            elif self.sheath_radio.isChecked():
                analysis_type = "Sheath analysis"
            elif self.lundquist_radio.isChecked():
                analysis_type = "Lundquist fit"
            else:
                analysis_type = "ForbMod"
            
            # Create temporary CDFDataManager to validate dates
            temp_manager = CDFDataManager()
            temp_manager.set_initial_params(self.selected_satellite, start_date, end_date, self.observer_input.text())
            
            # Validate date range and data availability
            if not temp_manager.validate_dates(start_date, end_date, analysis_type):
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Missing data files for selected date range.\nPlease ensure CDF files exist for all years in the range."
                )
                return
            
            self.callback(
                self.selected_satellite, 
                start_date, 
                end_date,
                self.observer_input.text(),
                analysis_type
            )
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid dates in YYYY/MM/DD format.")

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

class ForbModApp:
    def __init__(self, window_manager=None):
        # Get script directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize window manager
        if window_manager is None:
            self.window_manager = WindowManager(self.script_directory)
        else:
            self.window_manager = window_manager
        
        # Initialize managers
        self.data_manager = CDFDataManager()  # Use the new CDF-based data manager
        self.settings_manager = SettingsManager()
        
        self.start_window = None
        self.plot_window = None
        self.observer_name = None
        
        # Get the saved analysis type from settings
        self.analysis_type = self.settings_manager.get_analysis_type()
    
        self.show_start_window()
        
    def show_start_window(self):
        self.start_window = StartWindow(
            callback=self.on_start_window_complete,
            window_manager=self.window_manager)
        self.start_window.on_dates_changed = self.on_dates_changed 
        self.start_window.show()
            
    def on_start_window_complete(self, satellite, start_date, end_date, observer_name, analysis_type=None):
        try:
            logger.info(f"Processing start window completion for {satellite}")
            self.observer_name = observer_name
            
            # Store the analysis type exactly as received from the start window
            self.analysis_type = analysis_type
            
            # Save to settings manager for persistence
            if self.settings_manager and analysis_type:
                self.settings_manager.set_analysis_type(analysis_type)
            
            # Set initial parameters for new data manager
            self.data_manager.set_initial_params(satellite, start_date, end_date, observer_name)
            
            # Hide start window instead of closing it
            if self.start_window:
                self.start_window.hide()
                
            # Pass to show_plot_window explicitly
            self.show_plot_window()
            
        except Exception as e:
            logger.error(f"Error in start window completion: {str(e)}")
            QMessageBox.critical(self.start_window, "Error", 
                               f"Failed to process selection: {str(e)}")

    def on_plot_window_calculate(self, calc_results, fit_type, figure, results_dir):
        """Handle calculation results from the plot window"""
        try:
            # Save the settings
            if self.settings_manager:
                self.settings_manager.save_settings()
                
            # Display the fit window for ForbMod analysis
            if self.analysis_type == "ForbMod":
                self.fit_window = FitWindow(
                    sat=self.data_manager.satellite,
                    detector=self.data_manager.detector,
                    observer=self.observer_name,
                    calc_results=calc_results,
                    output_info={
                        'script_directory': self.script_directory,
                        'results_directory': results_dir,
                        'day': self.data_manager.start_date.strftime('%Y/%m/%d'),
                        'fit': fit_type.lower(),
                        'observer_name': self.observer_name,
                        'data_manager': self.data_manager,
                        'figure': figure
                    },
                    window_manager=self.window_manager
                )
                self.fit_window.show()
            elif self.analysis_type in ["In-situ analysis", "Sheath analysis"]:
                # For in-situ and sheath analyses, just show a success message
                QMessageBox.information(
                    self.plot_window,
                    "Success",
                    f"{self.analysis_type} results saved successfully"
                )
        except Exception as e:
            logger.error(f"Error handling plot window calculation: {str(e)}")
            QMessageBox.critical(self.plot_window, "Error", 
                               f"Failed to process calculation: {str(e)}")
    
    def on_dates_changed(self, new_start, new_end):
        try:
            # Save current window geometry before recreating
            if self.plot_window:
                self.window_manager.save_window_geometry(self.plot_window, 'plot_window')
                
            # Use update_dates method from new data manager
            self.data_manager.update_dates(new_start, new_end)
                
            if self.plot_window:
                self.plot_window.close()
                
            self.show_plot_window()
        except Exception as e:
            logger.error(f"Error changing dates: {str(e)}")
            QMessageBox.critical(self.plot_window, "Error", 
                               "Failed to update dates")
            
    def show_plot_window(self):
        try:
            
            # Ensure we have a valid analysis type, default to ForbMod if none
            if not self.analysis_type:
                self.analysis_type = "ForbMod"
                print("No analysis type set, defaulting to ForbMod")
            
            # Create the plot window with the explicit analysis type
            self.plot_window = PlotWindow(
                self.data_manager,
                observer_name=self.observer_name,
                on_calculate=self.on_plot_window_calculate,
                on_dates_changed=self.on_dates_changed,
                analysis_type=self.analysis_type,  # Pass the stored analysis type
                window_manager=self.window_manager,
                parent=self.start_window,
                sheath_analysis=(self.analysis_type == "Sheath analysis")  # Set based on actual analysis type
            )
            
            self.plot_window.show()
        except Exception as e:
            logger.error(f"Failed to show plot window: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(None, "Error", f"Failed to open plot window: {str(e)}")
            

def main():
    try:
        #silence the warning
        os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.xcb=false;qt.gui.icc=false;qt.qpa.fontdatabase=false'
        
        
        # Now create the QApplication (High DPI attributes already set)
        app = QApplication(sys.argv)

        # Apply PyQtDarkTheme auto selection. This provides a working "dark mode"
        qdarktheme.setup_theme("auto")
        
        # Create settings manager first
        script_directory = os.path.dirname(os.path.abspath(__file__))
        settings_manager = SettingsManager()
        
        # Create window manager with settings manager
        window_manager = WindowManager(script_directory, settings_manager)
        
        # Apply DPI scaling after application is created
        window_manager.setup_dpi_scaling()
        
        # Apply the icon to the application
        apply_app_icon(app)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Set up exception handling for Qt
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            logger.error("Uncaught exception", exc_info=(exctype, value, traceback))
            sys._excepthook(exctype, value, traceback)
        sys.excepthook = exception_hook
        
        # Initialize the application with window manager
        forb_mod_app = ForbModApp(window_manager)
        return app.exec_()
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        QMessageBox.critical(None, "Critical Error", 
                           "Application failed to start. Check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
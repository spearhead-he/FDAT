# main.py

# to handle matplotlib configs
from matplotlib_setup import configure_matplotlib
configure_matplotlib()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QLineEdit, 
    QFrame, QGridLayout, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont
from datetime import datetime, timedelta
import sys
import os
import logging
import math


# Configure logging once at the application level
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
from data_manager import DataManager
from settings_manager import SettingsManager


class StartWindow(QMainWindow):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.selected_satellite = None
        self.settings_manager = SettingsManager()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("ForbMod Analysis Tool")
        self.setMinimumSize(800, 600)
        
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
        
        # Analyze button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)
        layout.addWidget(self.analyze_btn)
        
        self.apply_styles()
        
    def create_header(self):
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(header)
        
        title = QLabel("ForbMod Analysis Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        description = QLabel("Analyze ICME/Forbush decrease events")
        description.setAlignment(Qt.AlignCenter)
        
        # Add observer name input
        observer_layout = QHBoxLayout()
        observer_label = QLabel("Observer:")
        self.observer_input = QLineEdit()
        self.observer_input.setText(self.settings_manager.get_observer_name())
        self.observer_input.textChanged.connect(self.save_observer_name)
        observer_layout.addWidget(observer_label)
        observer_layout.addWidget(self.observer_input)
        
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(observer_layout)
        return header

    def save_observer_name(self):
        """Save observer name when changed"""
        self.settings_manager.set_observer_name(self.observer_input.text())
        
    def create_date_section(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QGridLayout(frame)
        
        # Date inputs
        start_label = QLabel("Start Date:")
        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("YYYY/MM/DD")
        self.start_date_input.setText("1997/10/09")  # Default date
        
        end_label = QLabel("End Date:")
        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("YYYY/MM/DD")
        self.end_date_input.setText("1997/10/14")  # Default date
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        for days, text in [(-5, "◀ 5 days"), (-2, "◀ 2 days"), 
                          (2, "2 days ▶"), (5, "5 days ▶")]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, d=days: self.adjust_dates(d))
            nav_layout.addWidget(btn)
        
        layout.addWidget(start_label, 0, 0)
        layout.addWidget(self.start_date_input, 0, 1)
        layout.addWidget(end_label, 1, 0)
        layout.addWidget(self.end_date_input, 1, 1)
        layout.addLayout(nav_layout, 2, 0, 1, 2)
        
        return frame
        
    def create_satellite_section(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        label = QLabel("Select Satellite:")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(label)
        
        grid = QGridLayout()
        satellites = {
            "SolO": "Apr 2020 - Jul 2024",
            "OMNI": "Jan 2007 - Dec 2019",
            "ACE": "Sep 1997 - Dec 2022",
            "WIND": "Nov 1994 - Sep 2024",
            "Helios2": "Jan 1976 - Mar 1980",
            "Helios1": "Dec 1974 - Jun 1981"
        }
        
        self.satellite_buttons = {}
        for i, (sat, years) in enumerate(satellites.items()):
            btn = QPushButton(f"{sat}\n{years}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=sat: self.select_satellite(s))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(60)
            grid.addWidget(btn, i//3, i%3)
            self.satellite_buttons[sat] = btn
            
        layout.addLayout(grid)
        return frame
        
    def apply_styles(self):
        self.setStyleSheet("""
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
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QLabel {
                color: #333;
            }
        """)
        
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
            
            # Pass observer name along with other data    
            self.callback(
                self.selected_satellite, 
                start_date, 
                end_date,
                self.observer_input.text()
            )
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid dates in YYYY/MM/DD format.")

class ForbModApp:
    def __init__(self):
        self.data_manager = DataManager()
        self.settings_manager = SettingsManager()
        self.start_window = None
        self.plot_window = None
        self.observer_name = None
        self.show_start_window()
        
    def show_start_window(self):
        self.start_window = StartWindow(callback=self.on_start_window_complete)
        self.start_window.show()
        
    def on_start_window_complete(self, satellite, start_date, end_date, observer_name):
        try:
            logger.info(f"Processing start window completion for {satellite}")
            self.observer_name = observer_name
            
            # Set initial parameters including directory creation
            self.data_manager.set_initial_params(satellite, start_date, end_date, observer_name)
            
            if self.start_window:
                self.start_window.close()
                
            self.show_plot_window()
            
        except Exception as e:
            logger.error(f"Error in start window completion: {str(e)}")
            QMessageBox.critical(self.start_window, "Error", 
                               f"Failed to process selection: {str(e)}")
            
    def show_plot_window(self):
        try:
            self.plot_window = PlotWindow(
                self.data_manager,
                observer_name=self.observer_name,
                on_calculate=self.on_plot_window_calculate,
                on_dates_changed=self.on_dates_changed
            )
            self.plot_window.show()
        except Exception as e:
            logger.error(f"Failed to show plot window: {str(e)}")
            QMessageBox.critical(None, "Error", "Failed to open plot window")
            
    def on_plot_window_calculate(self, plot_data):
        try:
            self.data_manager.update_plot_data(plot_data)
            logger.info("Calculation completed")
        except Exception as e:
            logger.error(f"Error in calculation: {str(e)}")
            QMessageBox.critical(self.plot_window, "Error", 
                               "Failed to perform calculation")
            
    def on_dates_changed(self, new_start, new_end):
        try:
            self.data_manager.update_dates(new_start, new_end)
            if self.plot_window:
                self.plot_window.close()
            self.show_plot_window()
        except Exception as e:
            logger.error(f"Error changing dates: {str(e)}")
            QMessageBox.critical(self.plot_window, "Error", 
                               "Failed to update dates")

def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Set up exception handling for Qt
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            logger.error("Uncaught exception", exc_info=(exctype, value, traceback))
            sys._excepthook(exctype, value, traceback)
        sys.excepthook = exception_hook
        
        forb_mod_app = ForbModApp()
        return app.exec_()
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        QMessageBox.critical(None, "Critical Error", 
                           "Application failed to start. Check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
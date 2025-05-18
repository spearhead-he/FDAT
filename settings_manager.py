# settings_manager.py

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SettingsManager:
    def __init__(self):
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_settings.json')
        self.default_settings = {
            'observer_name': '',
            'analysis_type': 'ForbMod',
            # Add window settings defaults
            'window_settings': {
                'start_window': {
                    'size': [450, 650],
                    'position': ["center", "center"]
                },
                'plot_window': {
                    'size': [900, 700],
                    'position': ["center", "center"]
                },
                'fit_window': {
                    'size': [700, 500],
                    'position': ["center", "center"]
                },
                'dpi_factor': 1.0,
                'font_scale': 1.0
            }
        }
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from file or create with defaults if doesn't exist"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Deep merge with defaults to ensure all keys exist
                    settings = self._deep_merge_defaults(self.default_settings.copy(), settings)
                    return settings
            return self.default_settings.copy()
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return self.default_settings.copy()
    
    def _deep_merge_defaults(self, defaults, user_settings):
        """Recursively merge defaults with user settings, prioritizing user settings"""
        for key, default_value in defaults.items():
            # If key doesn't exist in user settings, add default
            if key not in user_settings:
                user_settings[key] = default_value
            # If both values are dictionaries, merge them recursively
            elif isinstance(default_value, dict) and isinstance(user_settings[key], dict):
                user_settings[key] = self._deep_merge_defaults(default_value, user_settings[key])
        return user_settings
            
    def save_settings(self):
        """Save current settings to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.settings_file)), exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
                
            logger.debug(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")

    def get_observer_name(self):
        """Get stored observer name"""
        return self.settings.get('observer_name', '')

    def set_observer_name(self, name):
        """Set and save observer name"""
        self.settings['observer_name'] = name
        self.save_settings()


    def set_analysis_type(self, analysis_type):
        """Set and save analysis type"""
        self.settings['analysis_type'] = analysis_type
        self.save_settings()

    def get_last_dates(self):
        """Get stored last used start and end dates"""
        start_date_str = self.settings.get('last_start_date', '')
        end_date_str = self.settings.get('last_end_date', '')
        
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
            except:
                pass
                
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, "%Y/%m/%d")
            except:
                pass
                
        return start_date, end_date
    
    def set_last_dates(self, start_date, end_date):
        """Set and save last used start and end dates"""
        if isinstance(start_date, datetime) and isinstance(end_date, datetime):
            self.settings['last_start_date'] = start_date.strftime("%Y/%m/%d")
            self.settings['last_end_date'] = end_date.strftime("%Y/%m/%d")
            self.save_settings()
    

    
    def set_window_settings(self, window_type, settings_dict):
        """Save settings for a specific window type"""
        if 'window_settings' not in self.settings:
            self.settings['window_settings'] = self.default_settings['window_settings']
        
        self.settings['window_settings'][window_type] = settings_dict
        self.save_settings()

    def set_window_state(self, window_type, state_dict):
        """Save complete window state including maximized flag"""
        if 'window_settings' not in self.settings:
            self.settings['window_settings'] = self.default_settings['window_settings']
        
        # Get current settings or create new dict if none exists
        if window_type not in self.settings['window_settings']:
            self.settings['window_settings'][window_type] = {}
        
        # Update with all state info
        self.settings['window_settings'][window_type].update(state_dict)
        
        # Force save to disk
        self.save_settings()
    

    def get_window_settings(self, window_type=None):
        """Get window settings, optionally for a specific window type"""
        window_settings = self.settings.get('window_settings', self.default_settings['window_settings'])
        if window_type:
            return window_settings.get(window_type, self.default_settings['window_settings'].get(window_type, {}))
        return window_settings
    

    def set_window_geometry(self, window_type, size, position):
        """Set window geometry settings"""
        window_settings = self.get_window_settings(window_type) or {}
        window_settings['size'] = size
        window_settings['position'] = position
        # Keep maximized state if it was set
        if 'maximized' not in window_settings:
            window_settings['maximized'] = False
        self.set_window_settings(window_type, window_settings)
    

    def get_dpi_factor(self):
        """Get DPI scaling factor"""
        window_settings = self.get_window_settings()
        return window_settings.get('dpi_factor', 1.0)
    
    def set_dpi_factor(self, factor):
        """Set DPI scaling factor"""
        window_settings = self.get_window_settings()
        window_settings['dpi_factor'] = factor
        self.settings['window_settings'] = window_settings
        self.save_settings()
    
    def get_font_scale(self):
        """Get font scaling factor"""
        window_settings = self.get_window_settings()
        return window_settings.get('font_scale', 1.0)
    
    def set_font_scale(self, scale):
        """Set font scaling factor"""
        window_settings = self.get_window_settings()
        window_settings['font_scale'] = scale
        self.settings['window_settings'] = window_settings
        self.save_settings()

    def get_analysis_type(self):
        """Get stored analysis type"""
        analysis_type = self.settings.get('analysis_type', 'ForbMod')
        return analysis_type
    
    def set_analysis_type(self, analysis_type):
        """Set and save analysis type"""
        self.settings['analysis_type'] = analysis_type
        self.save_settings()
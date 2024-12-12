# settings_manager.py

import os
import json
import logging

logger = logging.getLogger(__name__)

class SettingsManager:
    def __init__(self):
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_settings.json')
        self.default_settings = {
            'observer_name': ''
        }
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from file or create with defaults if doesn't exist"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            return self.default_settings.copy()
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return self.default_settings.copy()

    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")

    def get_observer_name(self):
        """Get stored observer name"""
        return self.settings.get('observer_name', '')

    def set_observer_name(self, name):
        """Set and save observer name"""
        self.settings['observer_name'] = name
        self.save_settings()
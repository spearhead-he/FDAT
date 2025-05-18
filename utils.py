# utils.py

import os
import json
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QFont, QFontDatabase
import logging

logger = logging.getLogger(__name__)

class WindowManager:
    def __init__(self, script_directory, settings_manager=None):
        self.script_directory = script_directory
        self.settings_manager = settings_manager  # Use the provided settings manager
        
        # Default window settings for reference
        self.default_settings = {
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
        
        # Create a settings property to maintain backward compatibility
        # This is what's missing and causing the error
        self.settings = {
            'dpi_factor': 1.0,
            'font_scale': 1.0
        }
        
        # If no settings manager provided, create our own properties
        if not self.settings_manager:
            # Try to find settings_manager
            try:
                from settings_manager import SettingsManager
                self.settings_manager = SettingsManager()
            except ImportError:
                logger.warning("SettingsManager couldn't be imported, using default settings")
        
        # Update our local settings from the settings_manager if available
        self._update_settings_from_manager()

    def _update_settings_from_manager(self):
        """Update local settings from settings_manager"""
        if self.settings_manager:
            # Update DPI and font factors
            self.settings['dpi_factor'] = self.settings_manager.get_dpi_factor()
            self.settings['font_scale'] = self.settings_manager.get_font_scale()
            
            # Update window settings
            for window_type in ['start_window', 'plot_window', 'fit_window']:
                window_settings = self.settings_manager.get_window_settings(window_type)
                if window_settings:
                    self.settings[window_type] = window_settings.copy()

    # Fix the setup_dpi_scaling method to handle both settings_manager and local settings
    def setup_dpi_scaling(self):
        """Optimized DPI scaling that applies uniform scaling to all UI elements"""
        try:
            # Get screen DPI
            screen = QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch()
            
            # Calculate scaling factor (standard DPI is 96)
            dpi_factor = dpi / 96.0
            
            # Limit to reasonable range and use minimal font scaling
            dpi_factor = max(0.8, min(1.5, dpi_factor))
            font_scale = 0.95 + 0.05 * (dpi_factor - 0.8) / 0.7  # 0.95-1.05 range
            
            # Save to settings
            if self.settings_manager:
                self.settings_manager.set_dpi_factor(dpi_factor)
                self.settings_manager.set_font_scale(font_scale)
            
            # Also update local settings for backward compatibility
            self.settings['dpi_factor'] = dpi_factor
            self.settings['font_scale'] = font_scale
            
            # Apply to application
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            
            app = QApplication.instance()
            if app:
                font = app.font()
                font.setPointSizeF(font.pointSizeF() * font_scale)
                app.setFont(font)
                
            # Set environment variables for Qt scaling
            os.environ["QT_SCALE_FACTOR"] = str(dpi_factor)
            
            logger.info(f"DPI scaling: {dpi_factor:.2f}, font scaling: {font_scale:.2f}")
            return dpi_factor, font_scale
            
        except Exception as e:
            logger.error(f"Error setting up DPI scaling: {str(e)}")
            return 1.0, 1.0

    def get_window_settings(self, window_type=None):
        """Get window settings, optionally for a specific window type"""
        if self.settings_manager:
            # Always use settings_manager when available
            window_settings = self.settings_manager.get_window_settings(window_type)
            if window_type and not window_settings:
                # Fall back to defaults if not found
                return self.default_settings.get(window_type, {})
            return window_settings
        
        # Fall back to default settings if no settings_manager
        return self.default_settings.get(window_type, {}) if window_type else self.default_settings
    
    
    def save_window_geometry(self, window, window_type):
        """Save window size and position, normalized by DPI factor, accounting for maximized window state"""
        try:
            dpi_factor = self.settings_manager.get_dpi_factor() if self.settings_manager else self.settings.get('dpi_factor', 1.0)
            
            # Check if window is maximized
            is_maximized = window.isMaximized()
            
            # For maximized windows, save the maximized state instead of actual dimensions
            if is_maximized:
                # When maximized, store special values to indicate maximized state
                window_state = {
                    'maximized': True,
                    'size': self.settings.get(window_type, {}).get('size', [900, 700]),  # Keep previous size
                    'position': self.settings.get(window_type, {}).get('position', [100, 100])  # Keep previous position
                }
            else:
                # Normal window - save actual size and position
                size = window.size()
                pos = window.pos()
                normalized_width = int(size.width() / dpi_factor)
                normalized_height = int(size.height() / dpi_factor)
                
                window_state = {
                    'maximized': False,
                    'size': [normalized_width, normalized_height],
                    'position': [pos.x(), pos.y()]
                }
                
            # Save to settings manager
            if self.settings_manager:
                # Only update local settings if different to prevent redundant saving
                current_state = self.settings.get(window_type, {})
                if (window_type not in self.settings or 
                    current_state.get('maximized') != window_state['maximized'] or
                    current_state.get('size') != window_state['size'] or
                    current_state.get('position') != window_state['position']):
                    
                    # Update local settings
                    if window_type not in self.settings:
                        self.settings[window_type] = {}
                        
                    self.settings[window_type].update(window_state)
                    
                    # Save to settings manager which writes to disk
                    self.settings_manager.set_window_state(window_type, window_state)
                
        except Exception as e:
            logger.error(f"Error saving window geometry: {str(e)}")

    
            
    def apply_window_geometry(self, window, window_type):
        """Apply saved size and position to window with DPI-aware dimensions, respecting maximized state"""
        try:
            # Get window settings with maximized state
            window_settings = self.get_window_settings(window_type)
            is_maximized = window_settings.get('maximized', False)
            
            # If window should be maximized, just store the flag for later
            # Don't apply any geometry changes yet
            if is_maximized:
                # Store the maximized state as a property on the window
                window.setProperty("should_maximize", True)
                logger.info(f"Window {window_type} will be maximized when shown")
                return
                
            # For non-maximized windows, proceed with normal geometry application
            # Get screen information
            screen = QApplication.primaryScreen()
            screen_geo = screen.availableGeometry()
            screen_width = screen_geo.width()
            screen_height = screen_geo.height()
            
            # Get default dimensions for this window type
            default_width, default_height = self.default_settings[window_type]['size']
            
            # Get scaling factor
            dpi_factor = self.settings_manager.get_dpi_factor() if self.settings_manager else self.settings.get('dpi_factor', 1.0)
            
            # Get saved size with fallback to default dimensions
            saved_size = window_settings.get('size', [default_width, default_height])
            
            # Apply DPI scaling to the dimensions but keep within screen limits
            width = min(int(saved_size[0] * dpi_factor), screen_width - 110)
            height = min(int(saved_size[1] * dpi_factor), screen_height - 110)


            # Special handling for plot window to ensure control panel visibility
            if window_type == 'plot_window':
                # Ensure minimum height for plot window
                minimum_height = 700  # This should be enough for all panels
                if height < minimum_height:
                    height = minimum_height
                    
                # Ensure window doesn't exceed screen height - 50px (for taskbar)
                max_height = screen_height - 50
                if height > max_height:
                    height = max_height
                    
                # Set the height constraint
                window.setMinimumHeight(minimum_height)
                
            # Apply final size
            window.resize(width, height)
            
            # Apply size
            window.resize(width, height)
            
            # Calculate center position as fallback
            center_x = (screen_width - width) // 2
            center_y = (screen_height - height) // 2
            
            # Check if we have a saved position that's valid
            position = window_settings.get('position', None)
            if (position and len(position) == 2 and 
                isinstance(position[0], (int, float)) and isinstance(position[1], (int, float))):
                # Check if position would keep window on screen
                if (0 <= position[0] <= screen_width - width and 
                    0 <= position[1] <= screen_height - height):
                    # Use saved position
                    x, y = position[0], position[1]
                else:
                    # Use center position if saved position is off-screen
                    x, y = center_x, center_y
            else:
                # No valid saved position, use center
                x, y = center_x, center_y
            
            # Apply position
            window.move(x, y)
            
            logger.info(f"Applied {window_type} geometry - size: {width}x{height}, position: {x},{y}")
            
        except Exception as e:
            logger.error(f"Error applying window geometry: {str(e)}")
            # Simple fallback
            try:
                window.resize(800, 600)
                window.move(100, 100)
            except Exception as inner_e:
                logger.error(f"Fallback positioning failed: {str(inner_e)}")

    
    def apply_font_scaling(self, widget):
        """Apply font scaling to a widget and all its children recursively"""
        try:
            font_scale = self.settings_manager.get_font_scale() if self.settings_manager else self.settings.get('font_scale', 1.0)
            self._scale_widget_fonts(widget, font_scale)
        except Exception as e:
            logger.error(f"Error applying font scaling: {str(e)}")
    
    def _scale_widget_fonts(self, widget, scale_factor):
        """Recursively scale fonts for widget and all children with safety checks"""
        # Scale font of current widget
        font = widget.font()
        original_size = font.pointSizeF()
        
        # Only scale if size is positive and not zero
        if original_size > 0:
            # Ensure scaled size is not zero or negative
            new_size = original_size * scale_factor
            if new_size > 0:
                font.setPointSizeF(new_size)
                widget.setFont(font)
        
        # Process all child widgets recursively
        for child in widget.findChildren(QWidget):
            if child.parent() == widget:  # Only direct children
                self._scale_widget_fonts(child, scale_factor)
    
   
    def get_scaled_style_sheet(self, base_style):
        """Return a style sheet with font sizes scaled according to DPI"""
        try:
            font_scale = self.settings_manager.get_font_scale() if self.settings_manager else self.settings.get('font_scale', 1.0)
            
            # Function to scale pixel values in style sheet
            def scale_pixels(match):
                value = int(match.group(1))
                scaled = int(value * font_scale)
                return f"{scaled}px"
            
            # Scale font-size, padding, margin, height values
            import re
            scaled_style = re.sub(r'font-size:\s*(\d+)px', lambda m: f"font-size: {int(int(m.group(1)) * font_scale)}px", base_style)
            scaled_style = re.sub(r'padding:\s*(\d+)px', lambda m: f"padding: {int(int(m.group(1)) * font_scale)}px", scaled_style)
            scaled_style = re.sub(r'margin:\s*(\d+)px', lambda m: f"margin: {int(int(m.group(1)) * font_scale)}px", scaled_style)
            scaled_style = re.sub(r'height:\s*(\d+)px', lambda m: f"height: {int(int(m.group(1)) * font_scale)}px", scaled_style)
            
            return scaled_style
            
        except Exception as e:
            logger.error(f"Error scaling style sheet: {str(e)}")
            return base_style

    def get_sized_font(self, size_type):
        """
        Get a font with the appropriate size for different UI elements.
        
        Args:
            size_type: String indicating the type ('title', 'header', 'normal')
            
        Returns:
            QFont object with appropriate size
        """
        # Base sizes
        sizes = {
            'title': 24,    # Large size for main title
            'header': 14,   # Medium size for section headers
            'normal': 11    # Normal size for regular text
        }
        
        # Get font scale factor
        font_scale = self.settings_manager.get_font_scale() if self.settings_manager else self.settings.get('font_scale', 1.0)
        
        # Get the base size
        base_size = sizes.get(size_type, sizes['normal'])
        
        # Apply scaling
        scaled_size = int(base_size * font_scale)
        
        # Create and return font
        font = QFont("Arial", scaled_size)
        if size_type in ['title', 'header']:
            font.setBold(True)
            
        return font
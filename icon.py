
# Icon generation for FDAT 

import os
from PyQt5 import QtCore, QtGui, QtSvg

def load_app_icon(version=8):
    """Load application icon from SVG file
    
    Args:
        version: Integer version number (not used when loading from file)
        
    Returns:
        QIcon: Application icon
    """
    try:
        import os
        from PyQt5 import QtGui, QtCore, QtSvg
        
        # Create an icon using QIcon
        app_icon = QtGui.QIcon()
        
        # Path to the SVG icon file
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon8.svg')
        
        if os.path.exists(icon_path):
            # For program panel/taskbar, we need to add multiple sizes
            renderer = QtSvg.QSvgRenderer(icon_path)
            
            # Create QPixmap images of different sizes for better OS integration
            for size in [16, 24, 32, 48, 64, 128, 256]:
                pixmap = QtGui.QPixmap(size, size)
                pixmap.fill(QtCore.Qt.transparent)  # Create transparent background
                painter = QtGui.QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                app_icon.addPixmap(pixmap)
                
            #print(f"Icon loaded from: {icon_path}")
            return app_icon
        else:
            print(f"Warning: Icon file not found at {icon_path}")
            return QtGui.QIcon()
            
    except Exception as e:
        print(f"Error loading app icon: {e}")
        return QtGui.QIcon()

def apply_app_icon(app, windows=None, version=8):
    """Apply icon to application and all specified window classes
    
    Args:
        app: QApplication instance
        windows: Optional list of window classes to set the icon for
        version: Icon version to use (not used when loading from file)
    """
    try:
        # Load the icon
        app_icon = load_app_icon(version)
        
        # Set application icon
        app.setWindowIcon(app_icon)
        
        # For Windows platform - set application ID for proper taskbar grouping
        import sys
        if sys.platform == 'win32':
            try:
                import ctypes
                app_id = 'FDAT.Application.Viewer'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            except Exception as e:
                print(f"Warning: Could not set Windows AppUserModelID: {e}")
        
        # Set icon for all specified window classes
        if windows:
            for window_class in windows:
                if window_class:
                    window_class.setWindowIcon(app_icon)
                    
    except Exception as e:
        print(f"Error applying app icon: {e}")



# For testing the icon directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
    
    app = QApplication(sys.argv)
    
    # Create test window
    window = QMainWindow()
    window.setWindowTitle("FDAT Icon Test")
    window.resize(400, 300)
    
    # Set icon
    icon = load_app_icon(version=8)
    window.setWindowIcon(icon)
    
    # Add a label to display info
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    label = QLabel("FDAT icon test - check the window title bar")
    layout.addWidget(label)
    window.setCentralWidget(central_widget)
    
    # Show window
    window.show()
    
    sys.exit(app.exec_())
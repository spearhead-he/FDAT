import matplotlib
import logging

def configure_matplotlib():
    """Configure matplotlib settings to avoid warnings and optimize for Qt"""
    # Use Qt5 backend
    matplotlib.use('Qt5Agg')
    
    # Disable font-related warnings
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # Configure matplotlib rcParams
    matplotlib.rcParams.update({
        # Use a specific font family
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        
        # Optimize figure settings
        'figure.dpi': 100,
        'figure.figsize': [8.0, 6.0],
        
        # Configure axes
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.grid.which': 'major',
        
        # Configure grid
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        
        # Configure ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Configure legend
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        
        # Disable toolbar
        'toolbar': 'None'
    })

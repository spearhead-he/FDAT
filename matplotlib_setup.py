import matplotlib
import logging
import os

def configure_matplotlib():
    """Configure matplotlib settings to avoid warnings and optimize for Qt"""
    # Use Qt5 backend
    matplotlib.use('Qt5Agg')
    
    # Disable font-related warnings completely
    logging.getLogger('matplotlib.font_manager').disabled = True
    
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
        'grid.alpha': 0.3,
        
        # Configure ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Configure legend
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        
        # Disable toolbar
        'toolbar': 'None'
    })
    
    # Suppress leapseconds warning
    os.environ['SPACEPY_LEAPSECS_WARN'] = 'False'
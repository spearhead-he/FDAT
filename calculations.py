# calculations.py

import numpy as np
from scipy.special import jn_zeros, j0
from sklearn.metrics import mean_squared_error
import logging
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CalculationManager:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def perform_calculations(self, regions, analysis_type="In-situ analysis"):
        """
        Main calculation function that works for all analysis types
        
        Args:
            regions: Dictionary containing analysis regions defined by the user
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary of calculation results
        """
        try:
            # Load analysis configuration from JSON
            analysis_config = self.load_analysis_config(analysis_type)
            if not analysis_config:
                logger.warning(f"No configuration found for analysis type: {analysis_type}")
                analysis_config = self._default_analysis_config(analysis_type)
            
            # Initialize results structure
            results = {
                'timestamps': {},
                'parameters': {},  # Use a single parameters dictionary for all results
                'coordinates': {},
                'has_gcr_data': False
            }

            # Get coordinates at main region start time
            if 'main' in regions and 'start' in regions['main']:
                results['coordinates'] = self.data_manager.get_coordinates(regions['main']['start'])
            else:
                # Fallback to default coordinates
                results['coordinates'] = self.data_manager.get_coordinates()
            
            # Process regions and add timestamps
            for region_name, region_data in regions.items():
                if 'start' in region_data and 'end' in region_data:
                    start_dt = region_data['start']
                    end_dt = region_data['end']
                    
                    # Add datetime values
                    results['timestamps'][f"{region_name}_start_dt"] = start_dt
                    results['timestamps'][f"{region_name}_end_dt"] = end_dt
                    
                    # Add DOY values
                    results['timestamps'][f"doy_{region_name}_start"] = self.data_manager.datetime_to_doy(start_dt)
                    results['timestamps'][f"doy_{region_name}_end"] = self.data_manager.datetime_to_doy(end_dt)
            
            # Process all configured calculations
            for calc in analysis_config.get('calculations', []):
                # Each calculation specifies: 
                # - data_key: what data to use
                # - operation: what operation to perform
                # - region: which region to use (defaults to 'main')
                # - result_key: where to store the result - name
                # - window_size: optional window size for edge/monent calculations (in minutes) - to avoid taking peak/low/nan values
                
                data_key = calc.get('data_key')
                operation = calc.get('operation')
                region_name = calc.get('region', 'main')
                result_key = calc.get('result_key')
                window_size = calc.get('window_size', 30)  # Default to 30 minutes for edge values
                
                # Skip if missing essential information
                if not data_key or not operation or not result_key:
                    logger.warning(f"Incomplete calculation specification: {calc}")
                    continue
                    
                # Skip if region doesn't exist
                if region_name not in regions:
                    logger.warning(f"Region {region_name} not defined for calculation {result_key}")
                    continue
                    
                # Get data from data manager
                data = self.data_manager.get_data(data_key)
                if data is None or len(data) == 0:
                    logger.warning(f"No data available for {data_key}")
                    continue
                    
                # Track GCR data availability
                if data_key == 'GCR':
                    results['has_gcr_data'] = True
                    
                # Perform calculation
                value = self.calculate_parameter(
                    data_key, 
                    operation, 
                    regions[region_name]['start'],
                    regions[region_name]['end'],
                    window_size=window_size,
                    data=data
                )
                
                # Store result
                results['parameters'][result_key] = value
            
            # Perform fit if configured
            fit_model = analysis_config.get('fit_model')
            if fit_model and results.get('has_gcr_data', False):
                # for Forbmod
                if fit_model == "bessel":
                    fit_result = self.perform_bessel_fit(regions, results)
                    results['fit'] = fit_result
                elif fit_model == "lundquist" and 'mo' in regions:  # Add Lundquist fit for MO region
                    # For Lundquist fit, show parameter dialog
                    try:
                        from lundquist_dialog import LundquistParamDialog
                        from PyQt5.QtWidgets import QApplication
                        
                        # Get main window (parent) for dialog
                        parent = QApplication.activeWindow()
                        param_dialog = LundquistParamDialog(parent)
                        
                        if param_dialog.exec_():
                            parameters = param_dialog.get_parameters()
                            
                            # Get data for MO region
                            mo_regions = {'main': regions['mo']}
                            if 'upstream' in regions:
                                mo_regions['upstream'] = regions['upstream']
                                
                            # Perform Lundquist fit
                            fit_result = self.perform_lundquist_fit(mo_regions, parameters)
                            results['fit'] = fit_result
                    except Exception as e:
                        logger.error(f"Error in Lundquist fit: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())

            # Ensure the results have all necessary entries
            if 'timestamps' not in results:
                results['timestamps'] = {}
                
            # Add DOY values from regions to timestamps
            if 'main' in regions:
                doy_start = self.data_manager.datetime_to_doy(regions['main']['start'])
                doy_end = self.data_manager.datetime_to_doy(regions['main']['end'])
                
                results['timestamps']['doy_start'] = doy_start
                results['timestamps']['doy_end'] = doy_end
                
                # If FD_min_DOY not calculated
                if 'FD_min_DOY' not in results['timestamps']:
                    results['timestamps']['FD_min_DOY'] = np.nan
            
            # Ensure velocities, magnetic and fd sections exist
            if 'velocities' not in results:
                results['velocities'] = {}
            if 'magnetic' not in results:
                results['magnetic'] = {}
            if 'fd' not in results:
                results['fd'] = {}
                
            # Move parameters into appropriate sections - maybe these groups won't be used
            for key, value in list(results['parameters'].items()):
                if key.startswith('B'):
                    results['magnetic'][key] = value
                elif key.startswith('v') or key == 'upstream_w':
                    results['velocities'][key] = value
                elif key.startswith('FD'):
                    results['fd'][key] = value
            
            return results
            
        except Exception as e:
            logger.error(f"Error in calculations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    
    def calculate_parameter(self, data_key, operation, start_dt, end_dt, window_size=30, data=None):
        """
        Calculate a parameter using a specified operation with xarray-based slicing
        
        Args:
            data_key: Key for data to use
            operation: Operation to perform
            start_dt: Start datetime
            end_dt: End datetime
            window_size: Window size in minutes for edge/moment calculations (peak, lead etc.)
            data: Optional data array (not used with xarray approach)
            
        Returns:
            Calculated parameter value
        """

        # Adjust window size for MAVEN which has lower resolution data
        if hasattr(self.data_manager, 'satellite') and self.data_manager.satellite == 'MAVEN':
            if operation in ['lead', 'leading_edge', 'trail', 'trailing_edge', 'center']:
                window_size = 240  # Use 4 hours for MAVEN
                
        try:
            # Get data for specific time range from data_manager
            time_range_data = self.data_manager.get_data_for_range((start_dt, end_dt))
            
            # Find which dataset contains our key
            dataset_found = False
            dataset_values = None
            dataset_times = None
            
            for dataset_type in ['mf', 'sw', 'gcr', 'gcr_secondary', 'coords']:
                if dataset_type in time_range_data and data_key in time_range_data[dataset_type]:
                    dataset_values = time_range_data[dataset_type][data_key]
                    dataset_times = time_range_data[dataset_type]['time']
                    dataset_found = True
                    break
                    
            if not dataset_found or dataset_values is None or len(dataset_values) == 0:
                logger.warning(f"No data found for {data_key} in time range")
                return np.nan
                    
            # Filter out NaN values
            valid_mask = ~np.isnan(dataset_values)
            valid_values = dataset_values[valid_mask]
            valid_times = dataset_times[valid_mask] if dataset_times is not None else None
            
            if len(valid_values) == 0:
                logger.warning(f"No valid data points for {data_key} in time range")
                return np.nan
            
            # Handle different operations
            if operation == 'min':
                # Special handling for GCR data - to calculate FD as percentage
                if data_key == 'GCR':
                    # Find first valid value for normalization
                    first_valid = valid_values[0]
                    
                    if first_valid != 0:  # Avoid division by zero
                        # Calculate normalized values as percentage
                        normalized_values = (valid_values - first_valid) / first_valid * 100
                        # Find minimum (which will be negative for a decrease)
                        min_value = np.nanmin(normalized_values)
                        # Return absolute value for FD amplitude
                        return abs(min_value)
                    else:
                        logger.warning("First GCR value is zero, cannot normalize")
                        return np.nan
                else:
                    # For non-GCR data, just return minimum
                    return np.nanmin(valid_values)
                    
            elif operation == 'max' or operation == 'peak':
                return np.nanmax(valid_values)
            elif operation == 'mean' or operation == 'average':
                return np.nanmean(valid_values)
            elif operation == 'median':
                return np.nanmedian(valid_values)
            elif operation == 'std' or operation == 'stdev':
                return np.nanstd(valid_values)
            elif operation == 'time_of_min':
                if valid_times is None:
                    return np.nan
                    
                # For GCR data, find time of minimum after normalization
                if data_key == 'GCR':
                    first_valid = valid_values[0]
                    if first_valid != 0:
                        normalized_values = (valid_values - first_valid) / first_valid * 100
                        min_idx = np.nanargmin(normalized_values)
                        min_time = valid_times[min_idx]
                        return self.data_manager.datetime_to_doy(min_time)
                    else:
                        logger.warning("First GCR value is zero, cannot normalize")
                        return np.nan
                else:
                    # For non-GCR data
                    min_idx = np.nanargmin(valid_values)
                    min_time = valid_times[min_idx]
                    return self.data_manager.datetime_to_doy(min_time)
                    
            elif operation == 'time_of_max':
                if valid_times is None:
                    return np.nan
                    
                max_idx = np.nanargmax(valid_values)
                max_time = valid_times[max_idx]
                return self.data_manager.datetime_to_doy(max_time)
                
            elif operation in ['lead', 'leading_edge', 'trail', 'trailing_edge', 'center']:
                # For parameters calculations at time moments, we need to average data from a small time window (30 min) - to avoid considering spikes, low numbers, nans, etc.
                window_delta = timedelta(minutes=window_size)
                
                if operation in ['lead', 'leading_edge']:
                    window_center = start_dt
                elif operation in ['trail', 'trailing_edge']:
                    window_center = end_dt
                else:  # center
                    window_center = start_dt + (end_dt - start_dt) / 2
                    
                window_start = window_center - window_delta
                window_end = window_center + window_delta
                
                # Get data for the specific window
                window_data = self.data_manager.get_data_for_range((window_start, window_end))
                
                # Find the data in the window
                for dataset_type in ['mf', 'sw', 'gcr', 'gcr_secondary']:
                    if dataset_type in window_data and data_key in window_data[dataset_type]:
                        window_values = window_data[dataset_type][data_key]
                        valid_window = window_values[~np.isnan(window_values)]
                        
                        if len(valid_window) > 0:
                            return np.nanmean(valid_window)
                
                return np.nan
                
            elif operation == 'upstream':
                # This is just the average over the whole region
                return np.nanmean(valid_values)
            else:
                logger.warning(f"Unknown operation: {operation}")
                return np.nan
                
        except Exception as e:
            logger.error(f"Error calculating parameter {data_key} with operation {operation}: {str(e)}")
            return np.nan
    

    def perform_bessel_fit(self, regions, results):
        """Perform Bessel function fit for ForbMod analysis using time-sliced data"""
        try:
            # Need main region
            if 'main' not in regions:
                logger.warning("Main region not defined for fit")
                return {}
            
            # Get time-sliced data
            start_dt = regions['main']['start']
            end_dt = regions['main']['end']
            time_range_data = self.data_manager.get_data_for_range((start_dt, end_dt))
            
            # Check if GCR data exists in the slice
            gcr_found = False
            gcr_values = None
            time_values = None
            
            if 'gcr' in time_range_data and 'GCR' in time_range_data['gcr']:
                gcr_values = time_range_data['gcr']['GCR']
                time_values = time_range_data['gcr']['time']
                gcr_found = True
            elif 'gcr_secondary' in time_range_data and 'GCR' in time_range_data['gcr_secondary']:
                gcr_values = time_range_data['gcr_secondary']['GCR']
                time_values = time_range_data['gcr_secondary']['time']
                gcr_found = True
                
            if not gcr_found or gcr_values is None or len(gcr_values) < 2:
                logger.warning("Not enough GCR data points for fit")
                return {
                    'r_timeseries': np.array([]),
                    'A_timeseries': np.array([]),
                    'FD_bestfit': np.nan,
                    'MSE': np.nan
                }
            
            # Find valid points
            valid_mask = ~np.isnan(gcr_values)
            valid_gcr = gcr_values[valid_mask]
            valid_times = time_values[valid_mask]
            
            if len(valid_gcr) < 2:
                logger.warning("Not enough valid GCR data points for fit")
                return {
                    'r_timeseries': np.array([]),
                    'A_timeseries': np.array([]),
                    'FD_bestfit': np.nan,
                    'MSE': np.nan
                }
            
            # Normalize times to -1:1 range for r parameter
            time_min = np.min(valid_times)
            time_max = np.max(valid_times)
            time_range = time_max - time_min
            
            if time_range <= np.timedelta64(0, 's'):
                logger.warning("Time range is zero or negative")
                return {
                    'r_timeseries': np.array([]),
                    'A_timeseries': np.array([]),
                    'FD_bestfit': np.nan,
                    'MSE': np.nan
                }
            
            # Convert time differences to float seconds to ensure proper normalization
            r_timeseries = np.array([
                -1.0 + 2.0 * ((t - time_min) / time_range) for t in valid_times
            ])
            
            # Normalize GCR values
            first_valid = valid_gcr[0]
            if first_valid != 0:  # Avoid division by zero
                A_timeseries = (valid_gcr - first_valid) / first_valid
            else:
                # If first value is zero, use a small baseline value
                logger.warning("First GCR value is zero, using small baseline instead")
                A_timeseries = valid_gcr - first_valid
            
            # Calculate best fit
            best_fit = self.find_best_bessel_fit(A_timeseries, r_timeseries)
            
            # Create fit result
            fit_data = {
                'r_timeseries': r_timeseries,
                'A_timeseries': A_timeseries,
                'FD_bestfit': best_fit['amplitude'],
                'MSE': best_fit['mse'],
                'best_fit_bessel': best_fit['curve'],
                'r': best_fit['r_points']
            }
            
            return fit_data
            
        except Exception as e:
            logger.error(f"Error in Bessel fit: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'r_timeseries': np.array([]),
                'A_timeseries': np.array([]),
                'FD_bestfit': np.nan,
                'MSE': np.nan
            }
    
    def find_best_bessel_fit(self, A_timeseries, r_timeseries):
        """Find the best-fit Bessel function"""
        try:
            # Filter NaN values
            mask = ~np.isnan(A_timeseries) & ~np.isnan(r_timeseries)
            A_clean = A_timeseries[mask]
            r_clean = r_timeseries[mask]
            
            if len(A_clean) < 2:
                return {
                    'amplitude': np.nan,
                    'mse': np.nan,
                    'curve': np.array([]),
                    'r_points': np.array([])
                }
            
            # Get the first zero of the Bessel function J0
            lambda1 = jn_zeros(0, 1)[0]
            
            min_error = np.inf
            best_min = None
            best_fit_bessel = None
            
            # Try different minimum values from the data
            for i in range(len(A_clean)):
                current_min = np.min(A_clean[i:])
                
                # Calculate Bessel function with current amplitude
                A_bessel = -j0(lambda1 * np.abs(r_clean)) * (-current_min)
                
                # Calculate error
                error = mean_squared_error(A_clean, A_bessel)
                
                # Update if better fit found
                if error < min_error:
                    min_error = error
                    best_fit_bessel = A_bessel
                    best_min = current_min
            
            # Generate smooth curve for final plot 
            # Create evenly spaced points from -1 to 1
            r_points = np.linspace(-1, 1, 100)
            
            # Calculate the Bessel function curve
            best_fit_curve = -j0(lambda1 * np.abs(r_points)) * (-best_min)
            
            return {
                'amplitude': round(abs(best_min) * 100, 2),  # Convert to percentage
                'mse': min_error,
                'curve': best_fit_curve,
                'r_points': r_points
            }
            
        except Exception as e:
            logger.error(f"Error finding best Bessel fit: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'amplitude': np.nan,
                'mse': np.nan,
                'curve': np.array([]),
                'r_points': np.array([])
            }

    def get_operation_config(self, op_name):
        """Get configuration for a specific operation"""
        try:
            # Try to get from JSON configuration
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'analysis-config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'operations' in config and op_name in config['operations']:
                        return config['operations'][op_name]
            
            # Return default configuration if not found from json file
            defaults = {
                'average': {'function': 'np.nanmean', 'round': 1},
                'mean': {'function': 'np.nanmean', 'round': 1},
                'median': {'function': 'np.nanmedian', 'round': 1},
                'stdev': {'function': 'np.nanstd', 'round': 1},
                'std': {'function': 'np.nanstd', 'round': 1},
                'min': {'function': 'np.nanmin', 'round': 2},
                'max': {'function': 'np.nanmax', 'round': 1},
                'peak': {'function': 'np.nanmax', 'round': 1},
                'lead': {'custom': True, 'round': 1},
                'trail': {'custom': True, 'round': 1},
                'center': {'custom': True, 'round': 1},
                'time_of_min': {'custom': True, 'round': 1}
            }
            
            return defaults.get(op_name, {})
            
        except Exception as e:
            logger.error(f"Error getting operation config: {str(e)}")
            return {}
    
    def get_data_for_timerange(self, data_key, start_dt, end_dt):
        """Get data for a specific key within a time range with proper time alignment"""
        try:
            # Try to get time-aligned data directly from data_manager if available
            if hasattr(self.data_manager, 'get_data_for_range'):
                data_dict = self.data_manager.get_data_for_range((start_dt, end_dt))
                
                # Look for the key in all datasets
                for dataset_type in ['mf', 'sw', 'gcr', 'gcr_secondary']:
                    if dataset_type in data_dict and data_key in data_dict[dataset_type]:
                        return data_dict[dataset_type][data_key]
            
            # Fallback to manual filtering
            data_values = self.data_manager.get_data(data_key)
            time_values = self.data_manager.get_data('time')
            
            if data_values is None or time_values is None or len(data_values) != len(time_values):
                return np.array([])
            
            # Convert datetime to numpy.datetime64
            start_np = np.datetime64(start_dt)
            end_np = np.datetime64(end_dt)
            
            # Filter data for the region
            in_region = (time_values >= start_np) & (time_values <= end_np)
            return data_values[in_region]
            
        except Exception as e:
            logger.error(f"Error getting data for {data_key}: {str(e)}")
            return np.array([])
    
    def perform_lundquist_fit(self, regions, parameters):
        """Perform Lundquist fit for the MO region
        
        Args:
            regions: Dictionary containing regions
            parameters: Dictionary with lundquist parameters
            
        Returns:
            Dictionary with fitting results
        """
        try:
            # Check if lmfit is installed - needed only for this fit
            try:
                import lmfit
            except ImportError:
                logger.error("lmfit package not found. Please install with 'pip install lmfit'")
                return {}
                
            # Import lundquist_fit module
            try:
                from lundquist import lundquist_fit
            except ImportError:
                logger.error("lundquist_fit.py module not found in current directory")
                return {}

            # Get parameters from dialog if not provided
            if parameters is None and show_dialog:
                try:
                    from lundquist.lundquist_dialog import LundquistParamDialog
                    from PyQt5.QtWidgets import QApplication
                    
                    # Get main window (parent) for dialog
                    parent = QApplication.activeWindow()
                    param_dialog = LundquistParamDialog(parent)
                    
                    if param_dialog.exec_():
                        parameters = param_dialog.get_parameters()
                    else:
                        # User canceled dialog
                        return {}
                except Exception as e:
                    logger.error(f"Error showing parameter dialog: {str(e)}")
                    return {}

            # If we still don't have parameters, return empty result
            if parameters is None:
                logger.warning("No parameters provided for Lundquist fit")
                return {}
            
            # Get data from data_manager
            data = self.data_manager.load_data()
            
            # Get the region times
            region_datetime = {
                'mo_start': regions['main']['start'],
                'mo_end': regions['main']['end']
            }
            
            if 'upstream' in regions:
                region_datetime['upstream_start'] = regions['upstream']['start']
                region_datetime['upstream_end'] = regions['upstream']['end']
            
            # Create output directory
            script_directory = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(
                script_directory,
                'OUTPUT',
                self.data_manager.satellite,
                'Sheath_analysis',
                regions['main']['start'].strftime('%Y_%m_%d'),
                'lundquist_fit'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Perform the fit
            result = lundquist_fit.perform_lundquist_fit(data, region_datetime, parameters, output_dir)
            
            # Convert to expected format for returning
            return {
                'lundquist_parameters': result['optimized_parameters'],
                'derived_parameters': result['derived_parameters'],
                'fit_curves': {
                    'btot': result['fit_data']['btot_fit'],
                    'br': result['fit_data']['br_fit'],
                    'bt': result['fit_data']['bt_fit'],
                    'bn': result['fit_data']['bn_fit']
                },
                'method': 'lundquist'
            }
            
        except Exception as e:
            logger.error(f"Error in Lundquist fit: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def get_section_for_data_key(self, data_key):
        """Determine which results section a data key belongs to"""
        if data_key in ['B', 'Bx', 'By', 'Bz', 'dB']:
            return 'magnetic'
        elif data_key in ['V', 'Vx', 'Vy', 'Vz']:
            return 'velocities'
        elif data_key in ['N', 'T', 'Beta']:
            return 'plasma'
        elif data_key in ['GCR', 'FD']:
            return 'fd'
        else:
            return 'other'  # Fallback section
    
    def get_result_key(self, data_key, op_name):
        """Generate default result key for a data key and operation"""
        try:
            # Default mappings if not found
            default_mappings = {
                'B': {
                    'average': 'BAvg', 'mean': 'BAvg', 'median': 'BMedian', 
                    'stdev': 'BStdev', 'std': 'BStdev', 'peak': 'BPeak', 'max': 'BPeak'
                },
                'dB': {
                    'average': 'dBAvg', 'mean': 'dBAvg'
                },
                'V': {
                    'average': 'vAvg', 'mean': 'vAvg', 'median': 'vMedian',
                    'stdev': 'vStdev', 'std': 'vStdev', 'peak': 'vPeak', 'max': 'vPeak',
                    'lead': 'vLead', 'trail': 'vTrail', 'center': 'v_center'
                },
                'GCR': {
                    'min': 'FD_obs', 'time_of_min': 'FD_min_DOY'
                }
            }
            
            # Check if we have a mapping
            if data_key in default_mappings and op_name in default_mappings[data_key]:
                return default_mappings[data_key][op_name]
            
            # Generate a standard key format
            op_suffix = op_name.capitalize()
            return f"{data_key}{op_suffix}"
            
        except Exception as e:
            logger.error(f"Error getting result key: {str(e)}")
            # Simple fallback
            return f"{data_key}_{op_name}"
    
    def get_coordinates(self, target_time=None):
        """
        Get spacecraft coordinates at a specific time (or default to start date)
        
        Args:
            target_time: Specific time to get coordinates for. If None, uses start_date
            
        Returns:
            Dictionary with distance, longitude, latitude values
        """
        coords = {
            'distance': np.nan,
            'longitude': np.nan,
            'latitude': np.nan
        }
        
        try:
            # Use provided target time or default to start date
            time_to_use = target_time if target_time is not None else self.start_date
            
            # Get data for 24-hour range around the target time
            buffer_hours = 12
            start_range = time_to_use - timedelta(hours=buffer_hours)
            end_range = time_to_use + timedelta(hours=buffer_hours)
            
            # Get data for this range
            data = self.get_data_for_range((start_range, end_range))
            
            if 'coords' in data and data['coords'] and 'time' in data['coords']:
                coord_time = data['coords']['time']
                
                # Find index of closest time to target
                target_np = np.datetime64(time_to_use)
                time_diffs = np.abs(coord_time - target_np)
                closest_idx = np.argmin(time_diffs)
                
                # Only use if within 3 hours
                if time_diffs[closest_idx] <= np.timedelta64(3, 'h'):
                    # Get values at this index
                    if 'dist' in data['coords'] and not np.isnan(data['coords']['dist'][closest_idx]):
                        coords['distance'] = float(data['coords']['dist'][closest_idx])
                    
                    if 'clon' in data['coords'] and not np.isnan(data['coords']['clon'][closest_idx]):
                        coords['longitude'] = float(data['coords']['clon'][closest_idx])
                    
                    if 'clat' in data['coords'] and not np.isnan(data['coords']['clat'][closest_idx]):
                        coords['latitude'] = float(data['coords']['clat'][closest_idx])
                        
                    logger.info(f"Found coordinates at {pd.Timestamp(coord_time[closest_idx])}: "
                               f"R={coords['distance']:.2f} AU, "
                               f"lon={coords['longitude']:.1f}°, "
                               f"lat={coords['latitude']:.1f}°")
            
            return coords
            
        except Exception as e:
            logger.error(f"Error getting coordinates: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return coords
    
    def load_analysis_config(self, analysis_type):
        """Load configuration for specified analysis type"""
        try:
            # Load from JSON file
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'analysis-config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'analysis_types' in config and analysis_type in config['analysis_types']:
                        return config['analysis_types'][analysis_type]
            
            # Return None if not found
            return None
            
        except Exception as e:
            logger.error(f"Error loading analysis config: {str(e)}")
            return None
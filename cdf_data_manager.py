import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
import time
import xarray as xr
from spacepy import pycdf
import pandas as pd
import json
import math
import glob

# Set up logger
logger = logging.getLogger(__name__)

class CDFDataManager:
    def __init__(self):
        self.satellite = None
        self.start_date = None
        self.end_date = None
        self.observer_name = None
    
        # Caching system
        self.file_cache = {}  # Cache for raw file data
        self.slice_cache = {}  # Cache for processed data
        self.max_cache_size = 20

        self.gcr_directory = {}
        self.secondary_gcr_directory = {}
        
        # Initialize empty dictionaries for satellite mappings
        self.detector = {}
        self.secondary_detector = {}
        self.gcr_mapping = {}
        self.secondary_gcr_mapping = {}
        self.ip_data_source = {}
            
        # Load mappings from JSON files
        self.variable_mappings = self.load_variable_mappings()
        self.load_satellite_mappings()

    def set_initial_params(self, satellite, start_date, end_date, observer_name):
        """Set initial parameters and clear cache if satellite changed"""
        # Clear cache only if satellite changes
        if self.satellite != satellite:
            self.file_cache = {}
            self.slice_cache = {}
            
        self.satellite = satellite
        self.start_date = start_date
        self.end_date = end_date
        self.observer_name = observer_name
        
        logger.info(f"Initial parameters set for {satellite} from {start_date} to {end_date}")

    def load_cdf_as_xarray(self, filename):
        """Load CDF file as xarray Dataset with time coordinate as index for better performance"""
        try:
            # Check cache first
            if filename in self.file_cache:
                logger.debug(f"Using cached xarray dataset for {os.path.basename(filename)}")
                return self.file_cache[filename]
                
            if not os.path.exists(filename):
                logger.warning(f"File not found: {filename}")
                return None
                
            logger.info(f"Loading CDF file as xarray: {os.path.basename(filename)}")
            start_time = time.time()
            
            # Open the CDF file
            with pycdf.CDF(filename) as cdf_file:
                # Find time variable
                time_var = None
                for var_name in cdf_file:
                    if 'EPOCH' in var_name.upper() or 'TIME' in var_name.upper():
                        time_var = var_name
                        break
                
                if not time_var:
                    logger.warning(f"No time variable found in {filename}")
                    return None
                
                # Read time data -  for indexing
                time_data = cdf_file[time_var][...]
                time_type = cdf_file[time_var].type()
                
                # Create data dictionary for xarray
                data_vars = {}
                
                # Process each variable
                for var_name in cdf_file:
                    if var_name != time_var:
                        try:
                            # Get variable data
                            var_data = cdf_file[var_name][...]
                            
                            # Handle multi-dimensional variables
                            if len(var_data.shape) > 1:
                                # If it's a vector variable with multiple components
                                if len(var_data.shape) > 1 and var_data.shape[1] <= 3:
                                    # Likely x, y, z components
                                    components = ['x', 'y', 'z'][:var_data.shape[1]]
                                    # Create a separate variable for each component
                                    for i, comp in enumerate(components):
                                        data_vars[f"{var_name}_{comp}"] = (['time'], var_data[:, i])
                                else:
                                    # For other multi-dimensional variables, use standard dimension names
                                    dims = ['time']
                                    coords = {}
                                    for i in range(1, len(var_data.shape)):
                                        dim_name = f"dim_{i}"
                                        dims.append(dim_name)
                                        coords[dim_name] = np.arange(var_data.shape[i])
                                    data_vars[var_name] = (dims, var_data, coords)
                            else:
                                # Regular 1D variable
                                data_vars[var_name] = (['time'], var_data)
                        except Exception as e:
                            logger.warning(f"Error processing variable {var_name}: {str(e)}")
                
                # Create xarray dataset with time as the index 
                ds = xr.Dataset(data_vars=data_vars)
                ds = ds.assign_coords(time=time_data)
                ds = ds.set_index(time='time')
                ds.attrs['filename'] = filename
                ds.attrs['title'] = os.path.basename(filename)
                ds.attrs['time_type'] = time_type  # Store time type in attributes
                
                # Copy variable attributes
                for var_name in cdf_file:
                    if var_name in ds:
                        if hasattr(cdf_file[var_name], 'attrs'):
                            for attr_name in cdf_file[var_name].attrs:
                                ds[var_name].attrs[attr_name] = cdf_file[var_name].attrs[attr_name]
                
                # Cache the dataset for future use
                self._add_to_file_cache(filename, ds)
                
                end_time = time.time()
                logger.info(f"Loaded {os.path.basename(filename)} as xarray in {end_time - start_time:.2f}s")
                
                return ds
                
        except Exception as e:
            logger.error(f"Error loading CDF as xarray: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _add_to_file_cache(self, key, value):
        """Add item to file cache with size management"""
        # If cache is full, remove least recently used item
        if len(self.file_cache) >= self.max_cache_size:
            # Get oldest key (first one added)
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]

        # Add new item
        self.file_cache[key] = value

    def get_time_slice(self, dataset, start_time, end_time):
        """Extract a time slice from an xarray Dataset with robust time selection handling nanosecond precision"""
        try:
            if dataset is None:
                logger.warning("Dataset is None")
                return None
            
            # Convert start_time and end_time to numpy.datetime64 if they're not already
            start_np = np.datetime64(start_time)
            end_np = np.datetime64(end_time)
            
            # Check if the dataset has a time index
            if 'time' in dataset.indexes:
                try:
                    # Try efficient time-based slicing with the index first
                    time_slice = dataset.sel(time=slice(start_np, end_np))
                    
                    if len(time_slice.time) > 0:
                        logger.debug(f"Time slice selected: {len(time_slice.time)} points (indexed)")
                        return time_slice
                except KeyError as e:
                    # If exact indexing fails, try a more forgiving approach
                    logger.debug(f"Exact time indexing failed: {str(e)}. Trying alternative method.")
                    
                    # Try manual selection with boolean masking
                    try:
                        # This handles nanosecond precision differences
                        mask = (dataset.time >= start_np) & (dataset.time <= end_np)
                        time_slice = dataset.isel(time=mask)
                        
                        if len(time_slice.time) > 0:
                            logger.debug(f"Time slice selected: {len(time_slice.time)} points (boolean mask)")
                            return time_slice
                    except Exception as mask_err:
                        logger.debug(f"Boolean mask selection failed: {str(mask_err)}")
            
            # Fallback for datasets without time index or if the above methods failed
            if 'time' in dataset.coords:
                
                # Reset the index if we have one but previous methods failed
                if 'time' in dataset.indexes:
                    dataset = dataset.reset_index('time')
                
                # Try with numpy datetime comparison (most robust for precision differences)
                try:
                    # Convert all times to int64 nanosecond values for comparison
                    time_values = dataset.time.values.astype('datetime64[ns]').astype(np.int64)
                    start_ns = np.datetime64(start_np, 'ns').astype(np.int64)
                    end_ns = np.datetime64(end_np, 'ns').astype(np.int64)
                    
                    # Create mask using integer comparison (avoids precision issues)
                    mask = (time_values >= start_ns) & (time_values <= end_ns)
                    time_slice = dataset.isel(time=mask)
                    
                    if len(time_slice.time) > 0:
                        logger.debug(f"Time slice selected: {len(time_slice.time)} points (nanosecond integer comparison)")
                        return time_slice
                except Exception as ns_err:
                    logger.debug(f"Nanosecond integer comparison failed: {str(ns_err)}")
                    
                # Final fallback: convert to pandas and use its time handling
                try:
                    import pandas as pd
                    # Get time as pandas series for more flexible datetime handling
                    times = pd.Series(dataset.time.values)
                    mask = (times >= start_time) & (times <= end_time)
                    indices = np.where(mask)[0]
                    
                    if len(indices) > 0:
                        time_slice = dataset.isel(time=indices)
                        logger.debug(f"Time slice selected: {len(time_slice.time)} points (pandas fallback)")
                        return time_slice
                except Exception as pd_err:
                    logger.debug(f"Pandas fallback failed: {str(pd_err)}")
            
            logger.warning(f"No data points found in time range {start_time} to {end_time}")
            return None
                
        except Exception as e:
            logger.error(f"Error in get_time_slice: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    
    def load_data(self):
        """Load data for specified satellite and time period using pattern matching for variable names"""
        try:
            # Create a cache key for the full result
            cache_key = f"{self.satellite}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
            
            # Return cached result if available
            if cache_key in self.slice_cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.slice_cache[cache_key]
            
            # Initialize data structure
            data = {
                'mf': {},      # Magnetic field data
                'sw': {},      # Solar wind data
                'gcr': {},     # Primary GCR data
                'gcr_secondary': {},  # Secondary GCR data
                'coords': {}  # satellite location
            }
            
            # Get the years needed for this time range
            years = self._get_years_needed()
            logger.info(f"Loading data for years: {years}")
            
            # Load variable mappings if not already loaded
            if not hasattr(self, 'variable_mappings') or not self.variable_mappings:
                self.variable_mappings = self.load_variable_mappings()
                
            # Clear merged files tracking for this load operation
            self._merged_files = set()
            
            # Process each data type
            for data_type in ['mf', 'sw', 'gcr', 'gcr_secondary', 'coords']:
                # Get all files for this data type
                all_files = []
                for year in years:
                    is_secondary = (data_type == 'gcr_secondary')
                    type_to_use = 'gcr' if data_type.startswith('gcr') else data_type
                    files = self.find_data_files(type_to_use, year, is_secondary=is_secondary)
                    all_files.extend(files)
                
                patterns = {}
                if data_type == 'gcr_secondary':
                    # For secondary GCR, we need to use GCR patterns but with some modifications
                    if 'default_patterns' in self.variable_mappings and 'gcr' in self.variable_mappings['default_patterns']:
                        gcr_patterns = self.variable_mappings['default_patterns']['gcr']
                        
                        # Use GCR_additional for secondary GCR if available
                        if 'GCR_additional' in gcr_patterns:
                            patterns['GCR'] = gcr_patterns['GCR_additional']
                        else:
                            # Fallback to regular GCR patterns
                            patterns['GCR'] = gcr_patterns.get('GCR', [])
                elif 'default_patterns' in self.variable_mappings and data_type in self.variable_mappings['default_patterns']:
                    patterns = self.variable_mappings['default_patterns'][data_type]
                    
                # Add user patterns if available
                if 'user_patterns' in self.variable_mappings and data_type in self.variable_mappings['user_patterns']:
                    for var_name, user_patterns in self.variable_mappings['user_patterns'][data_type].items():
                        if var_name in patterns:
                            patterns[var_name].extend(user_patterns)
                        else:
                            patterns[var_name] = user_patterns
                
                # Process all files for this data type
                for file_path in all_files:
                    # Skip if we've already processed this exact file+data_type combination
                    file_type_key = f"{file_path}_{data_type}"
                    if file_type_key in self._merged_files:
                        continue
                    
                    # Mark as processed
                    self._merged_files.add(file_type_key)
                    
                    # Load as xarray
                    dataset = self.load_cdf_as_xarray(file_path)
                    if dataset is None:
                        continue
                        
                    # Get time slice for our date range
                    time_slice = self.get_time_slice(dataset, self.start_date, self.end_date)
                    if time_slice is None or len(time_slice.time) == 0:
                        continue
                    
                    # Extract variables based on patterns
                    extracted_data = {'time': time_slice.time.values}
                    
                    for target_var, pattern_list in patterns.items():
                        # Check all variables in the dataset against patterns
                        found_match = False
                        for pattern in pattern_list:
                            for var_name in time_slice.data_vars:
                                # ONLY use exact matching 
                                if pattern.upper() == var_name.upper():
                                    # Found an exact match - extract the data
                                    extracted_data[target_var] = time_slice[var_name].values
                                    found_match = True
                                    logger.debug(f"Exact match found: {var_name} → {target_var}")
                                    break
                            if found_match:
                                break
                    
                    # Only add the dataset if we found more than just time
                    if len(extracted_data) > 1:
                        if len(data[data_type]) == 0:
                            # First data for this type
                            data[data_type] = extracted_data
                        else:
                            # Merge with existing data
                            self._merge_datasets(data[data_type], extracted_data)
            
            # Calculate beta if we have both MF and SW data and Beta isn't already present
            if (data['mf'] and 'B' in data['mf'] and 
                data['sw'] and 'N' in data['sw'] and 'T' in data['sw'] and
                'Beta' not in data['sw']):
                self._calculate_beta(data)
                
            # Calculate expected temperature if we have velocity data
            if 'sw' in data and 'V' in data['sw'] and 'T_exp' not in data['sw']:
                self._calculate_expected_temperature(data)
            
            # Cache the combined result
            if len(self.slice_cache) >= self.max_cache_size * 2:
                # Remove oldest item
                oldest_key = next(iter(self.slice_cache))
                del self.slice_cache[oldest_key]
                
            self.slice_cache[cache_key] = data
            
            logger.info(f"Data loaded for {self.satellite} from {self.start_date} to {self.end_date}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'mf': {}, 'sw': {}, 'gcr': {}, 'gcr_secondary': {}, 'coords': {}}
    
    def find_data_files(self, data_type, year, is_secondary=False):
        """Find data files for a specific type and year"""
        data_dir = os.path.dirname(os.path.abspath(__file__))
        found_files = []
        
        if data_type in ['mf', 'sw']:
            # Get the actual IP data source name (fallback to self.satellite if not defined)
            ip_source = self.ip_data_source.get(self.satellite, self.satellite)
            
            # Find the variables in the specific files
            patterns = [
                f"{ip_source}_{data_type}_{year}.cdf",  # Specific file for mf or sw
                f"{ip_source}_ip_{year}.cdf",           # if merged dataset
                f"{ip_source}_combined_{year}.cdf"  # if merged dataset      
            ]
            
            # MF and SW can be in merged datasets - optimize the search then not to open it twice
            for pattern in patterns:
                # Use ip_source for BOTH the directory name and the file pattern
                path = os.path.join(data_dir, f"data/IP/{ip_source}/{pattern}")
                if os.path.exists(path):
                    # Mark as merged file if it matches merged file patterns
                    is_merged = "ip_" in pattern or "combined_" in pattern or "all_" in pattern
                    if is_merged:
                        if not hasattr(self, '_merged_files'):
                            self._merged_files = set()
                        self._merged_files.add(path)
                    found_files.append(path)              
                    
        elif data_type == 'gcr':
            # Choose directory based on primary or secondary
            if is_secondary:
                # Secondary GCR data (if available)
                if self.satellite in self.secondary_gcr_directory:
                    source_directory = self.secondary_gcr_directory.get(self.satellite)
                else:
                    return []  # No secondary source defined
            else:
                # Primary GCR data
                if self.satellite in self.gcr_directory:
                    source_directory = self.gcr_directory.get(self.satellite)
                else:
                    return []  # No primary source defined
            
            # Use the same patterns for both primary and secondary
            patterns = [
                f"{source_directory}_gcr_{year}.cdf",
                f"{source_directory}_{year}.cdf"
            
            ]
            
            # Try both possible GCR data directories
            for pattern in patterns:
                path = os.path.join(data_dir, f"data/GCR/{source_directory}/{pattern}")
                if os.path.exists(path):
                    found_files.append(path)
                    break  # Found a file in this directory, no need to check others
        
        elif data_type == 'coords':
            # Get the actual IP data source name for coordinates
            ip_source = self.ip_data_source.get(self.satellite, self.satellite)
            
            # Look for coordinate files in the IP directory
            search_dir = os.path.join(data_dir, f"data/IP/{ip_source}")
            pattern = os.path.join(search_dir, "*position*.cdf")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                found_files.append(matching_files[0])  # Just take the first matching file
        
        return found_files
        
    
    def _process_gcr_files(self, file_list, target_dict):
        """Process GCR files and store data in the target dictionary"""
        time_values = []
        gcr_values = []
        
        # Get GCR variable patterns from the variable mappings JSON
        gcr_patterns = []
        if hasattr(self, 'variable_mappings') and self.variable_mappings and 'default_patterns' in self.variable_mappings:
            if 'gcr' in self.variable_mappings['default_patterns']:
                # Get all patterns for GCR variables
                for var_key, patterns in self.variable_mappings['default_patterns']['gcr'].items():
                    gcr_patterns.extend(patterns)
        
        # Add user patterns if available
        if hasattr(self, 'variable_mappings') and self.variable_mappings and 'user_patterns' in self.variable_mappings:
            if 'gcr' in self.variable_mappings['user_patterns']:
                for var_key, patterns in self.variable_mappings['user_patterns']['gcr'].items():
                    gcr_patterns.extend(patterns)
        
        # If no patterns found in JSON, use defaults
        if not gcr_patterns:
            gcr_patterns = ['GCR', 'COUNT_RATE', 'COUNTS', 'RATE', 'FLUX', 'DOSE_RATE']
        
        for file_path in file_list:
            dataset = self.load_cdf_as_xarray(file_path)
            if dataset is None:
                continue
                
            time_slice = self.get_time_slice(dataset, self.start_date, self.end_date)
            if time_slice is None or len(time_slice.time) == 0:
                continue
                
            # Find GCR variable using patterns from JSON
            gcr_var = None
            
            # Try each pattern
            for pattern in gcr_patterns:
                for var_name in time_slice.data_vars:
                    if var_name != 'time' and pattern in var_name.upper():
                        gcr_var = var_name
                        break
                if gcr_var:
                    break
            
            # If no match, use first non-time variable
            if not gcr_var and len(time_slice.data_vars) > 0:
                for var_name in time_slice.data_vars:
                    if var_name != 'time':
                        gcr_var = var_name
                        break
            
            # Log what we found
            if gcr_var:
                logger.debug(f"Using variable '{gcr_var}' from {os.path.basename(file_path)}")
                time_values.extend(time_slice.time.values)
                gcr_values.extend(time_slice[gcr_var].values)
            else:
                logger.warning(f"No suitable GCR variable found in {os.path.basename(file_path)}")
        
        # Sort and store data if we have any
        if time_values:
            times = np.array(time_values)
            values = np.array(gcr_values)
            
            sort_indices = np.argsort(times)
            target_dict['time'] = times[sort_indices]
            target_dict['GCR'] = values[sort_indices]
            logger.info(f"Processed {len(times)} GCR data points")
        else:
            logger.info("No GCR data found for the specified time range")
                

    def _calculate_beta(self, data):
        """Calculate plasma beta from B, N, and T using xarray"""
        try:
            # Check if we have an xarray Dataset or a dictionary
            if not isinstance(data, dict):
                logger.warning("Expected dictionary data structure, received {type(data)}")
                return
                
            # Check if required data is available
            if ('mf' not in data or 'B' not in data['mf'] or 
                'sw' not in data or 'N' not in data['sw'] or 'T' not in data['sw']):
                logger.warning("Missing required data for beta calculation")
                return
                
            # Get the datasets
            mf_data = data['mf']
            sw_data = data['sw']
            
            # Create time-aligned datasets if possible
            if 'time' in mf_data and 'time' in sw_data:
                # Find common time points using numpy intersect1d
                mf_times = mf_data['time']
                sw_times = sw_data['time']
                
                # Calculate beta array
                beta_values = np.full_like(sw_data['N'], np.nan, dtype=float)
                
                # Define time threshold for matching points (seconds) - mostly for similar to minute resolution
                time_threshold = np.timedelta64(50, 's')
                
                # For each solar wind time point, find matching B value
                for i, sw_time in enumerate(sw_times):
                    # Skip invalid data points
                    if np.isnan(sw_data['N'][i]) or np.isnan(sw_data['T'][i]):
                        continue
                        
                    # Find closest B measurement
                    time_diffs = np.abs(mf_times - sw_time)
                    min_idx = np.argmin(time_diffs)
                    
                    # Only use if within threshold
                    if time_diffs[min_idx] <= time_threshold:
                        b_value = mf_data['B'][min_idx]
                        
                        # Skip invalid or zero B values
                        if np.isnan(b_value) or b_value == 0:
                            continue
                        
                        # Calculate beta: Beta = (T * constant + offset) * N / B²
                        beta = ((sw_data['T'][i] * 4.16 / 1e5) + 5.34) * sw_data['N'][i] / (b_value**2)
                        beta_values[i] = beta
                
                # Add to data dictionary
                data['sw']['Beta'] = beta_values
                logger.info(f"Calculated beta with time alignment")
                
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    ######################## the function is updated with dependance on R
    def _calculate_expected_temperature(self, data):
        """Calculate expected temperature based on solar wind velocity and heliocentric distance
        
        Formulas used:
        1. For 0.95 <= R <= 1.1 AU from Lopex (1987):
           - v < 500 km/s: Texp = ((0.031 * v - 5.100) ** 2) * 10^3
           - v >= 500 km/s: Texp = (0.51 * v - 142) * 10^3
        
        2. For R > 1.1 AU from Richardson (2013) based on Ulysses data:
           - Texp = (502 * v - 1.26e5) / R
           
        3. For R < 0.99 AU from Lopez & Freeman (1986):
           - v < 500 km/s: Texp = ((0.031 * v - 4.4) ** 2) * 10^3 / R
           - v >= 500 km/s: Texp = (0.77 * v - 265) * 10^3 / R
        """
        try:
            # Check if velocity data is available
            if 'sw' not in data or 'V' not in data['sw']:
                logger.warning("Missing velocity data for T_exp calculation")
                return
            
            # Get velocity data
            velocity = data['sw']['V']
            
            # Initialize result array
            t_exp = np.full_like(velocity, np.nan, dtype=float)
            
            # Get distance at the start of the window
            dist = 1.0  # Default to 1 AU
            if 'coords' in data and 'dist' in data['coords'] and len(data['coords']['dist']) > 0:
                # Use the first valid distance value
                valid_distances = data['coords']['dist'][~np.isnan(data['coords']['dist'])]
                if len(valid_distances) > 0:
                    dist = valid_distances[0]
                    logger.info(f"Using distance R = {dist:.2f} AU for expected temperature calculations")
                else:
                    logger.info(f"No valid distance values found, using default R = 1.0 AU")
            else:
                logger.info(f"No distance data available, using default R = 1.0 AU")
            
            # Calculate temperature for each velocity value
            for i, v in enumerate(velocity):
                if np.isnan(v):
                    continue
                
                # Apply appropriate formula based on distance
                if 0.95 <= dist <= 1.1:
                    # Near-Earth formula (0.95-1.1 AU)
                    if v < 500:
                        t_exp[i] = ((0.031 * v - 5.1) ** 2) * 10**3
                    else:
                        t_exp[i] = (0.51 * v - 142) * 10**3
                elif dist > 1.1:
                    # Beyond Earth formula (> 1 AU)
                    t_exp[i] = (502 * v - 1.26e5) / dist
                else:
                    # Inner heliosphere formula (< 0.95 AU)
                    if v < 500:
                        t_exp[i] = ((0.031 * v - 4.4) ** 2) * 10**3 / dist
                    else:
                        t_exp[i] = (0.77 * v - 265) * 10**3 / dist
            
            # Add calculated T_exp to solar wind dataset
            data['sw']['T_exp'] = t_exp
            
        except Exception as e:
            logger.error(f"Error calculating T_exp: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def update_dates(self, new_start, new_end):
        """Update date range efficiently without clearing file cache"""
        logger.info(f"Updating date range from {self.start_date}-{self.end_date} to {new_start}-{new_end}")
        
        # Save previous dates
        old_start = self.start_date
        old_end = self.end_date
        
        # Update dates
        self.start_date = new_start
        self.end_date = new_end
        
        # Clear slice cache but keep file cache
        # We only clear relevant slices to save memory
        keys_to_remove = []
        for key in self.slice_cache:
            if key.startswith(f"{self.satellite}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.slice_cache[key]
            
        logger.info(f"Cleared slice cache for previous date range")

    def _get_years_needed(self):
        """Get list of years needed for the selected period"""
        years = set()
        current_date = self.start_date
        while current_date <= self.end_date:
            years.add(current_date.year)
            current_date += timedelta(days=1)
        return sorted(list(years))
    
    def validate_dates(self, start_date, end_date, analysis_type="ForbMod"):
        """Check if required data files exist for specified date range"""
        try:
            # Calculate years needed
            temp_start = start_date
            temp_end = end_date
            years = set()
            current_date = temp_start
            while current_date <= temp_end:
                years.add(current_date.year)
                current_date += timedelta(days=1)
            years = sorted(list(years))
            
            # Check for required files
            for year in years:
                # Magnetic field data (required for all analysis types)
                mf_files = self.find_data_files('mf', year)
                if not mf_files:
                    logger.warning(f"Missing magnetic field data for {year}")
                    self.reset_state()
                    return False
                
                # Solar wind data (required for all analysis types)
                sw_files = self.find_data_files('sw', year)
                if not sw_files:
                    logger.warning(f"Missing solar wind data for {year}")
                    self.reset_state()
                    return False
                
                # GCR data (crucial only for ForbMod, only required for others)
                if analysis_type == "ForbMod":
                    gcr_files = self.find_data_files('gcr', year)
                    if not gcr_files:
                        logger.warning(f"Missing GCR data for {year} (needed for ForbMod analysis)")
                        self.reset_state()
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating dates: {str(e)}")
            self.reset_state()
            return False

    def get_data(self, key, hourly=False):
        """Get specific data variable"""
        try:
            # Load data if not already loaded
            data_dict = self.load_data()
            if not data_dict:
                return Nones
                
            # Search for key in all datasets
            for dataset_name, dataset in data_dict.items():
                if key in dataset:
                    return dataset[key]
            
            # Key not found
            logger.warning(f"Key '{key}' not found in any dataset")
            return None
            
        except Exception as e:
            logger.error(f"Error getting data for key {key}: {str(e)}")
            return None

    def get_data_for_range(self, datetime_range):
        """Get data for a specific datetime range
        
        Args:
            datetime_range: Tuple (min_datetime, max_datetime) with datetime objects
                    
        Returns:
            Dictionary with data for that specific range
        """
        try:
            min_date, max_date = datetime_range
            
            # Convert Python datetime to numpy.datetime64 for consistent comparison
            min_date_np = np.datetime64(min_date)
            max_date_np = np.datetime64(max_date)
            
            # Create cache key
            cache_key = f"{self.satellite}_timerange_{min_date.isoformat()}_{max_date.isoformat()}"
            
            # Check if we have this range cached
            if cache_key in self.slice_cache:
                return self.slice_cache[cache_key]
            
            # Get all data
            full_data = self.load_data()
            if not full_data:
                return {
                    'mf': {},
                    'sw': {},
                    'gcr': {},
                    'gcr_secondary': {},
                    'coords': {}
                }
            
            # Create filtered copy
            filtered_data = {
                'mf': {},
                'sw': {},
                'gcr': {},
                'gcr_secondary': {},
                'coords': {}
            }
            
            # For each dataset
            for dataset_type in ['mf', 'sw', 'gcr', 'gcr_secondary', 'coords']:
                # Check if dataset_type exists in full data
                if dataset_type not in full_data:
                    continue
                    
                dataset = full_data[dataset_type]
                if 'time' not in dataset or len(dataset['time']) == 0:
                    continue
                    
                # Get time data
                time_data = dataset['time']
                
                # Convert min/max dates to the same type as time_data for comparison
                if isinstance(time_data, np.ndarray) and time_data.dtype.kind == 'M':
                    # Use numpy datetime64 comparison
                    mask = (time_data >= min_date_np) & (time_data <= max_date_np)
                else:
                    # Fallback to regular comparison (shouldn't happen but just in case)
                    try:
                        mask = (time_data >= min_date) & (time_data <= max_date)
                    except TypeError:
                        # If that fails too, skip this dataset
                        print(f"Cannot filter {dataset_type}: incompatible types")
                        continue
                
                # Skip if no data in range
                if not np.any(mask):
                    continue
                    
                # Copy filtered data
                filtered_data[dataset_type]['time'] = time_data[mask]
                
                # Copy all other variables
                for var_name, var_data in dataset.items():
                    if var_name != 'time' and len(var_data) == len(time_data):
                        filtered_data[dataset_type][var_name] = var_data[mask]
            
            # Cache the result
            if len(self.slice_cache) >= self.max_cache_size * 2:
                # Remove oldest item
                oldest_key = next(iter(self.slice_cache))
                del self.slice_cache[oldest_key]
                        
            self.slice_cache[cache_key] = filtered_data
            
            return filtered_data
                
        except Exception as e:
            logger.error(f"Error getting data for range: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'mf': {},
                'sw': {},
                'gcr': {},
                'gcr_secondary': {},
                'coords': {}
            }
        
    def create_output_directory(self, date=None, fit_type=None):
        """Create and return path for specific fit type"""
        try:
            fit_type = fit_type or 'test'
            
            # Use the provided date or default to start_date
            date_to_use = date or self.start_date
            date_str = date_to_use.strftime('%Y_%m_%d')
            
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'OUTPUT',
                self.satellite,
                date_str
            )
            
            fit_dir = os.path.join(base_dir, fit_type.lower())
            
            # Create the directory if it doesn't exist
            os.makedirs(fit_dir, exist_ok=True)
            
            logger.info(f"Created output directory: {fit_dir}")
            return fit_dir
                
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            raise

    def datetime_to_doy(self, dt, continuous_across_years=False, reference_year=None):
        """
        Convert datetime to day of year (fractional) with support for change of DOY across years
        
        Args:
            dt: datetime value (numpy.datetime64, datetime.datetime, or numeric DOY)
            continuous_across_years: If True, DOY continues past 365/366 for dates in years after reference_year
            reference_year: Base year for continuous counting. If None, uses self.start_date.year
            
        Returns:
            float: DOY with fractional part
        """
        try:
            # Use the reference year for continuous counting if specified
            if reference_year is None and continuous_across_years and hasattr(self, 'start_date'):
                reference_year = self.start_date.year
                
            # Handle numpy.datetime64 objects
            if hasattr(dt, 'dtype') and np.issubdtype(dt.dtype, np.datetime64):
                # Convert numpy.datetime64 to Python datetime
                dt_obj = pd.Timestamp(dt).to_pydatetime()
                
                # Standard DOY calculation
                day_of_year = dt_obj.timetuple().tm_yday
                fraction = (dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second) / 86400.0
                doy = day_of_year + fraction
                
                # If continuous counting is enabled, add days for prior years
                if continuous_across_years and reference_year is not None and dt_obj.year > reference_year:
                    for year in range(reference_year, dt_obj.year):
                        # Add days in this year (366 for leap years, 365 for non-leap years)
                        doy += 366 if self.is_leap_year(year) else 365
                
                return doy
            
            # Regular Python datetime
            elif isinstance(dt, datetime):
                # Standard DOY calculation
                day_of_year = dt.timetuple().tm_yday
                fraction = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
                doy = day_of_year + fraction
                
                # If continuous counting is enabled, add days for prior years
                if continuous_across_years and reference_year is not None and dt.year > reference_year:
                    for year in range(reference_year, dt.year):
                        # Add days in this year (366 for leap years, 365 for non-leap years)
                        doy += 366 if self.is_leap_year(year) else 365
                
                return doy
            
            # If it's already a numeric value (like a DOY), return as is
            elif isinstance(dt, (int, float)):
                return float(dt)
            
            else:
                logger.warning(f"Unhandled datetime type {type(dt)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in datetime_to_doy: {str(e)}")
            return 0.0

    def is_leap_year(self, year):
        """Check if a year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    # Now update the doy_to_datetime function to handle continuous DOY values
    def doy_to_datetime(self, year, doy, continuous_across_years=False):
        """Convert day of year to datetime with support for continuous DOY
        
        Args:
            year: Base year
            doy: Day of year (can be >365/366 if continuous_across_years is True)
            continuous_across_years: If True, handle DOY values >365/366 by adding years
            
        Returns:
            datetime object
        """
        try:
            # Check for NaN
            if doy is None or (isinstance(doy, float) and math.isnan(doy)):
                logger.warning("Cannot convert NaN to datetime")
                return None
                
            # Handle continuous DOY values that span multiple years
            if continuous_across_years and doy > 366:
                # Start with base year
                current_year = year
                remaining_doy = doy
                
                # Subtract days for each year until we get to the correct year
                while remaining_doy > (366 if self.is_leap_year(current_year) else 365):
                    days_in_year = 366 if self.is_leap_year(current_year) else 365
                    remaining_doy -= days_in_year
                    current_year += 1
                
                # Use the remaining DOY with the calculated year
                year = current_year
                doy = remaining_doy
            
            # Standard DOY to datetime conversion
            base_date = datetime(year, 1, 1)
            int_doy = int(doy)
            fraction = doy - int_doy
            hours = int(fraction * 24)
            minutes = int((fraction * 24 - hours) * 60)
            seconds = int(((fraction * 24 - hours) * 60 - minutes) * 60)
            
            return base_date + timedelta(days=int_doy-1, hours=hours, minutes=minutes, seconds=seconds)
            
        except Exception as e:
            logger.error(f"Error in doy_to_datetime: {str(e)}")
            return None


    def load_satellite_mappings(self):
        """Load satellite mappings from JSON file"""
        try:
            # Define the path to the mappings file
            mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'satellite_mappings.json')
          
            with open(mapping_path, 'r') as f:
                        mappings = json.load(f)
            
            # Process each satellite
            for sat_name, sat_data in mappings['satellites'].items():
                # Get detector mapping
                if 'detector' in sat_data:
                    self.detector[sat_name] = sat_data['detector']
                
                # Get secondary detector mapping
                if 'secondary_detector' in sat_data:
                    self.secondary_detector[sat_name] = sat_data['secondary_detector']
                
                # Get IP data source
                if 'ip_data_source' in sat_data:
                    self.ip_data_source[sat_name] = sat_data['ip_data_source']
                
                # Get GCR directory - no longer need tuple with dummy value
                if 'gcr_directory' in sat_data:
                    self.gcr_directory[sat_name] = sat_data['gcr_directory']
                
                # Get secondary GCR directory - no longer need tuple with dummy value
                if 'secondary_gcr_directory' in sat_data:
                    self.secondary_gcr_directory[sat_name] = sat_data['secondary_gcr_directory']
            
            # Store satellite_dates 
            self.satellite_dates = {}
            for sat_name, sat_data in mappings['satellites'].items():
                if 'date_range' in sat_data:
                    self.satellite_dates[sat_name] = sat_data['date_range']
            
        except Exception as e:
            logger.error(f"Unexpected error in loading satellite_mappings: {str(e)}")

    def load_variable_mappings(self):
        """Load variable pattern mappings from JSON configuration file"""
        try:
            # Define the path to the mappings file
            mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'variable_mappings.json')
            
            # Load mappings from file
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
                return mappings
            else:
                logger.warning(f"Variable mappings file not found: {mapping_path}")
                return self._default_variable_mappings()
        
        except Exception as e:
            logger.error(f"Error loading variable mappings: {str(e)}")
            return self._default_variable_mappings()
            
    
    def _default_variable_mappings(self):
        """Provide default variable mappings if json file was not found - just in case"""
        return {
            "default_patterns": {
                "mf": {
                    "B": ["B_MAG", "B", "BTOT"],
                    "Bx": ["BR", "B_x", "BX"],
                    "By": ["BT", "B_y", "BY"],
                    "Bz": ["BN", "B_z", "BZ"],
                    "dB": ["B_fluct", "dB"]
                },
                "sw": {
                    "N": ["dens", "N", "density"],
                    "V": ["V_MAG", "V", "speed"],
                    "Vx": ["VR", "V_x"],
                    "Vy": ["VT", "V_y"],
                    "Vz": ["VN", "V_z"],
                    "T": ["T", "temp", "temperature"],
                    "T_exp": ["T_exp", "expected_temp"],
                    "Beta": ["Beta", "beta"]
                },
                "gcr": {
                    "GCR": ["GCR", "counts", "rate"]
                }
            },
            "user_patterns": {}
        }

    def get_coordinates(self, target_time=None):
        """
        Get spacecraft coordinates at a specific time
        
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
            
            # Round to nearest hour since coordinates are hourly
            hour_rounded = time_to_use.replace(minute=0, second=0, microsecond=0)
            
            # Load data if not already loaded
            data_dict = self.load_data()
            if not data_dict or 'coords' not in data_dict or not data_dict['coords']:
                logger.warning("No coordinate data available")
                return coords
                
            # Get coordinate data
            coord_data = data_dict['coords']
            if 'time' not in coord_data or len(coord_data['time']) == 0:
                logger.warning("No time values in coordinate data")
                return coords
                
            # Convert time_to_use to np.datetime64 for comparison
            target_np = np.datetime64(hour_rounded)
            
            # Find exact match (since coordinates are exactly hourly)
            exact_matches = np.where(coord_data['time'] == target_np)[0]
            
            if len(exact_matches) > 0:
                # Exact match found - use it
                idx = exact_matches[0]
                
                # Get coordinate values
                if 'dist' in coord_data:
                    coords['distance'] = float(coord_data['dist'][idx])
                
                if 'clon' in coord_data:
                    coords['longitude'] = float(coord_data['clon'][idx])
                
                if 'clat' in coord_data:
                    coords['latitude'] = float(coord_data['clat'][idx])
                    
                logger.debug(f"Found exact coordinate match at {hour_rounded}")
            else:
                # Find closest time if no exact match
                time_diffs = np.abs(coord_data['time'] - target_np)
                closest_idx = np.argmin(time_diffs)
                
                # Get coordinate values from closest time
                if 'dist' in coord_data:
                    coords['distance'] = float(coord_data['dist'][closest_idx])
                
                if 'clon' in coord_data:
                    coords['longitude'] = float(coord_data['clon'][closest_idx])
                
                if 'clat' in coord_data:
                    coords['latitude'] = float(coord_data['clat'][closest_idx])
                    
                logger.debug(f"Using closest coordinate time match (diff: {time_diffs[closest_idx]})")
            
            return coords
            
        except Exception as e:
            logger.error(f"Error getting coordinates: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return coords


    def _merge_datasets(self, existing_data, new_data):
        """
        Merge new data into existing data, handling time sorting properly
        """
        # Return immediately if no new data
        if not new_data or 'time' not in new_data or len(new_data['time']) == 0:
            return
            
        # If existing data is empty, just copy the new data
        if not existing_data or 'time' not in existing_data or len(existing_data['time']) == 0:
            existing_data.update(new_data)
            return
        
        # Combine and sort times
        all_times = np.concatenate([existing_data['time'], new_data['time']])
        sort_indices = np.argsort(all_times)
        existing_data['time'] = all_times[sort_indices]
        
        # Merge all other variables
        for var_name, values in new_data.items():
            if var_name == 'time':
                continue  # Already handled time
                
            if var_name in existing_data:
                # Concatenate and sort
                combined_values = np.concatenate([existing_data[var_name], values])
                existing_data[var_name] = combined_values[sort_indices]
            else:
                # New variable - create properly sized array and fill with values
                new_array = np.full(len(all_times), np.nan, dtype=float)
                
                # Find indices where the new data's times appear in the merged timeline
                indices = np.searchsorted(all_times, new_data['time'])
                
                # Put values in the right spots
                for i, idx in enumerate(indices):
                    if idx < len(new_array):
                        new_array[idx] = values[i]
                
                # Sort according to the merged timeline
                existing_data[var_name] = new_array[sort_indices]

    def reset_state(self):
        """Reset cache to ensure clean operation"""
        # Clear caches
        self.file_cache = {}
        self.slice_cache = {}
        
        # Reset tracking for merged files
        if hasattr(self, '_merged_files'):
            self._merged_files = set()
        
        logger.info("CDFDataManager state has been reset")
# calculations.py

import numpy as np
from scipy.special import jn_zeros, j0
from sklearn.metrics import mean_squared_error
import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)

class CalculationManager:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def doy_to_hour_index(self, doy):
        """Convert DOY to hour index"""
        start_doy = self.data_manager.start_date.timetuple().tm_yday
        doy_int = int(doy)
        hour = int((doy - doy_int) * 24)
        hour_index = (doy_int - start_doy) * 24 + hour
        print(f"  Start DOY: {start_doy}")
        return hour_index

    def perform_calculations(self, doy_start, doy_end, upstream_start, upstream_end):
        """Main calculation function with proper GCR data handling"""
        try:
            logger.info("Starting calculations...")
            print(f"Selected region in DOY: {doy_start:.2f} to {doy_end:.2f}")
            
            # Initialize gcr_data before try block
            gcr_data = None
            
            try:
                # Get data arrays
                v_data = self.data_manager.get_data('V')
                b_data = self.data_manager.get_data('B')
                gcr_data = self.data_manager.get_data('GCR', hourly=True)
            except Exception as e:
                logger.warning(f"Error getting data: {str(e)}")
                
            # Check for GCR data availability
            has_gcr_data = gcr_data is not None and len(gcr_data) > 0 and not np.all(np.isnan(gcr_data))
            
            if not has_gcr_data:
                logger.warning("No GCR data available. Proceeding with ICME parameter calculations only.")
                print("Warning: No GCR data available. Only ICME parameters will be calculated.")
            
            # Calculate speeds if data is available
            speeds = {
                'vLead': np.nan,
                'vTrail': np.nan,
                'v_center': np.nan,
                'upstream_w': np.nan,
                'vAvg': np.nan,
                'vMedian': np.nan,
                'vStdev': np.nan,
                'vPeak': np.nan
            }
            
            if v_data is not None and len(v_data) > 0 and not np.all(np.isnan(v_data)):
                speeds = self.calculate_speeds(v_data, doy_start, doy_end, upstream_start, upstream_end)
    
            # Calculate magnetic parameters if data is available
            magnetic = {
                'BPeak': np.nan,
                'BAvg': np.nan,
                'BMedian': np.nan,
                'BStdev': np.nan
            }
            
            if b_data is not None and len(b_data) > 0 and not np.all(np.isnan(b_data)):
                magnetic = self.calculate_magnetic_params(b_data, doy_start, doy_end)
    
            # Initialize GCR-related parameters with NaN
            fd_params = {
                'reference_value': np.nan,
                'fd_amplitude': np.nan,
                'fd_min_doy': np.nan,
                'fd_data': np.array([])
            }
            
            fit_data = {
                'r_timeseries': np.array([]),
                'A_timeseries': np.array([]),
                'best_fit_bessel': None,
                'r': None,
                'FD_bestfit': None,
                'MSE': None,
                'details': {
                    'points_total': 0,
                    'points_valid': 0,
                    'points_nan': 0
                }
            }
    
            # Calculate GCR parameters only if data is available
            if has_gcr_data:
                start_doy = int(doy_start)
                end_doy = int(doy_end)
                start_hour = int((doy_start - start_doy) * 24)
                end_hour = int((doy_end - end_doy) * 24)
                t_hour = (start_doy - self.data_manager.start_date.timetuple().tm_yday) * 24 + start_hour
                z_hour = (end_doy - self.data_manager.start_date.timetuple().tm_yday) * 24 + end_hour
                
                fd_params = self.calculate_fd_parameters(gcr_data, t_hour, z_hour)
                fit_data = self.prepare_fit_data(gcr_data, t_hour, z_hour, speeds['vLead'], speeds['vTrail'])
    
            return {
                'timestamps': {
                    'doy_start': round(doy_start, 1),
                    'doy_end': round(doy_end, 1),
                    'FD_min_DOY': fd_params['fd_min_doy']
                },
                'velocities': speeds,
                'magnetic': magnetic,
                'fd': {
                    'FD_obs': fd_params['fd_amplitude']
                },
                'fit': fit_data,
                'coordinates': {
                    'distance': self.data_manager.distance,
                    'longitude': self.data_manager.longitude,
                    'latitude': self.data_manager.latitude
                },
                'has_gcr_data': has_gcr_data
            }
    
        except Exception as e:
            logger.error(f"Error in calculations: {str(e)}")
            raise
    
    def prepare_fit_data(self, gcr_data, t_hour, z_hour, vLead, vTrail):
        """Prepare data for fitting with proper NaN handling and independence from speed data"""
        try:
            # Get hourly GCR data for fitting
            hourly_data = self.data_manager.get_data('GCR', hourly=True)
            forbush = hourly_data[t_hour:z_hour+1].copy()
            
            print(f"Number of total GCR points: {len(forbush)}")
            print(f"Number of NaN points: {np.sum(np.isnan(forbush))}")
            
            # Create time points array
            num_points = len(forbush)
            r_points = np.linspace(-1, 1, num_points)
            
            # Filter out NaN values while preserving corresponding r points
            valid_mask = ~np.isnan(forbush)
            valid_forbush = forbush[valid_mask]
            valid_r_points = r_points[valid_mask]
            
            print(f"Number of valid points after NaN filtering: {len(valid_forbush)}")
            
            if len(valid_forbush) < 2:
                raise ValueError(f"Not enough valid points for fit. Need at least 2, got {len(valid_forbush)}")
                
            # Normalize relative to first valid point
            first_valid = valid_forbush[0]
            forbush_norm = (valid_forbush - first_valid) / first_valid
    
            return {
                'r_timeseries': valid_r_points,
                'A_timeseries': forbush_norm,
                'best_fit_bessel': None,
                'r': None,
                'FD_bestfit': None,
                'MSE': None,
                'details': {
                    'points_total': len(forbush),
                    'points_valid': len(valid_forbush),
                    'points_nan': np.sum(np.isnan(forbush))
                }
            }
    
        except Exception as e:
            logger.error(f"Error preparing fit data: {str(e)}")
            raise

    def calculate_average(self, index, data, window=60):
        """Calculate average around an index with proper NaN handling"""
        try:
            # Handle NaN index
            if np.isnan(index):
                return np.nan
                
            # Convert to integer safely
            index = int(np.floor(index))
            
            # Set window boundaries
            start_idx = max(0, index - window//2)
            end_idx = min(len(data), index + window//2)
            
            # Extract window data
            window_data = data[start_idx:end_idx]
            valid_data = window_data[~np.isnan(window_data)]
            
            return np.mean(valid_data) if len(valid_data) > 0 else np.nan
            
        except Exception as e:
            logger.error(f"Error calculating average: {str(e)}")
            return np.nan
    
    def calculate_speeds(self, v_data, doy_start, doy_end, upstream_start, upstream_end):
        """Calculate all speed-related parameters with proper NaN handling"""
        try:
            # Convert DOY to indices safely
            start_index = np.nan
            end_index = np.nan
            
            try:
                start_doy = self.data_manager.start_date.timetuple().tm_yday
                start_index = (doy_start - start_doy) * 24 * 60
                end_index = (doy_end - start_doy) * 24 * 60
            except Exception as e:
                logger.warning(f"Error converting DOY to indices: {str(e)}")
            
            # Calculate center index
            center = np.nan
            if not (np.isnan(start_index) or np.isnan(end_index)):
                center = (start_index + end_index) // 2
            
            # Calculate speeds with NaN handling
            vLead = round(self.calculate_average(start_index, v_data)) if not np.isnan(start_index) else np.nan
            vTrail = round(self.calculate_average(end_index, v_data)) if not np.isnan(end_index) else np.nan
            v_center = round(self.calculate_average(center, v_data)) if not np.isnan(center) else np.nan
            
            # Calculate upstream speed
            upstream_w = self.calculate_upstream(upstream_start, upstream_end, v_data)
            
            # Calculate statistics
            vPeak, vAvg, vMedian, vStdev = self.calculate_stats(
                start_index if not np.isnan(start_index) else 0,
                end_index if not np.isnan(end_index) else len(v_data),
                v_data
            )
            
            return {
                'vLead': vLead,
                'vTrail': vTrail,
                'v_center': v_center,
                'upstream_w': upstream_w,
                'vAvg': vAvg,
                'vMedian': vMedian,
                'vStdev': vStdev,
                'vPeak': vPeak
            }
            
        except Exception as e:
            logger.error(f"Error calculating speeds: {str(e)}")
            raise
    
    def calculate_upstream(self, start, end, data):
        """Calculate upstream speed average with NaN handling"""
        try:
            # Handle NaN inputs
            if np.isnan(start) or np.isnan(end):
                return np.nan
                
            # Convert to integers safely
            start_idx = int(np.floor(start))
            end_idx = int(np.ceil(end))
            
            if start_idx >= end_idx or start_idx < 0:
                logger.warning("Invalid upstream window indices")
                return np.nan
                
            # Get data slice and filter NaNs
            data_slice = data[start_idx:end_idx]
            valid_data = data_slice[~np.isnan(data_slice)]
            
            # Calculate mean if we have valid data
            if len(valid_data) > 0:
                return round(np.mean(valid_data))
            
            return np.nan
            
        except Exception as e:
            logger.error(f"Error calculating upstream speed: {str(e)}")
            return np.nan
    
    def calculate_stats(self, start, end, data):
        """Calculate statistical parameters with proper NaN handling"""
        try:
            # Handle NaN inputs
            if np.isnan(start) or np.isnan(end):
                return np.nan, np.nan, np.nan, np.nan
                
            # Convert to integers safely
            s_idx = int(np.floor(max(0, min(start, len(data)-1))))
            e_idx = int(np.ceil(max(0, min(end, len(data)))))
            
            # Get data slice and filter NaNs
            window_data = data[s_idx:e_idx]
            valid_data = window_data[~np.isnan(window_data)]
            
            if len(valid_data) == 0:
                return np.nan, np.nan, np.nan, np.nan
                
            return (
                round(np.max(valid_data), 1),
                round(np.mean(valid_data), 1),
                round(np.median(valid_data), 1),
                round(np.std(valid_data), 1)
            )
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return np.nan, np.nan, np.nan, np.nan

    def calculate_magnetic_params(self, b_data, doy_start, doy_end):
        """Calculate magnetic field parameters"""
        try:
            start_index = (doy_start - self.data_manager.start_date.timetuple().tm_yday) * 24 * 60
            end_index = (doy_end - self.data_manager.start_date.timetuple().tm_yday) * 24 * 60
            
            BPeak, BAvg, BMedian, BStdev = self.calculate_stats(start_index, end_index, b_data)
            
            return {
                'BPeak': BPeak,
                'BAvg': BAvg,
                'BMedian': BMedian,
                'BStdev': BStdev
            }

        except Exception as e:
            logger.error(f"Error calculating magnetic parameters: {str(e)}")
            raise

    def calculate_fd_parameters(self, gcr_data, t_hour, z_hour):
        """Calculate FD parameters"""
        try:
            # Get hourly GCR data
            hourly_data = self.data_manager.get_data('GCR', hourly=True)
            
            # Get GCR data window
            window_data = hourly_data[t_hour:z_hour+1]
            valid_indices = np.where(~np.isnan(window_data))[0]
            
            if len(valid_indices) == 0:
                raise ValueError("No valid GCR data in selected window")
            
            # Get reference point (first valid value in window)
            first_valid_idx = valid_indices[0]
            reference_value = window_data[first_valid_idx]
            
            # Calculate FD parameters
            fd_data = (window_data - reference_value) / reference_value * 100
            FDmin = np.nanmin(fd_data)
            min_idx = valid_indices[np.argmin(fd_data[valid_indices])]
            
            # Calculate FD amplitude and DOY
            fd_amplitude = abs(FDmin)
            fd_min_doy = round((first_valid_idx + min_idx)/24 + 
                              self.data_manager.start_date.timetuple().tm_yday, 1)
    
            return {
                'reference_value': reference_value,
                'fd_amplitude': round(fd_amplitude, 2),
                'fd_min_doy': fd_min_doy,
                'fd_data': fd_data
            }
    
        except Exception as e:
            logger.error(f"Error calculating FD parameters: {str(e)}")
            raise



    
    def calculate_fd_parameters(self, gcr_data, t_hour, z_hour):
        """Calculate FD parameters with NaN handling"""
        try:
            # Get hourly GCR data
            hourly_data = self.data_manager.get_data('GCR', hourly=True)
            
            # Get GCR data window
            window_data = hourly_data[t_hour:z_hour+1]
            
            # Find first valid value (non-NaN)
            valid_indices = np.where(~np.isnan(window_data))[0]
            
            if len(valid_indices) == 0:
                raise ValueError("No valid GCR data in selected window")
            
            # Get reference point (first valid value in window)
            first_valid_idx = valid_indices[0]
            reference_value = window_data[first_valid_idx]
            
            # Calculate FD parameters using only valid data
            fd_data = np.full_like(window_data, np.nan)
            fd_data[valid_indices] = (window_data[valid_indices] - reference_value) / reference_value * 100
            
            # Find minimum in valid data
            valid_fd = fd_data[valid_indices]
            FDmin = np.nanmin(valid_fd)
            min_idx = valid_indices[np.argmin(valid_fd)]
            
            # Calculate FD amplitude and DOY
            fd_amplitude = abs(FDmin)
            fd_min_doy = round((first_valid_idx + min_idx)/24 + 
                              self.data_manager.start_date.timetuple().tm_yday, 1)
    
            return {
                'reference_value': reference_value,
                'fd_amplitude': round(fd_amplitude, 2),
                'fd_min_doy': fd_min_doy,
                'fd_data': fd_data
            }
    
        except Exception as e:
            logger.error(f"Error calculating FD parameters: {str(e)}")
            raise


    
    def find_best_bessel_fit(self, A_timeseries, r_timeseries):
        """
        Find the best-fit Bessel function for FD modeling
        
        Args:
            A_timeseries: Normalized FD amplitude data
            r_timeseries: Normalized radial distance data
            
        Returns:
            tuple: (best_fit_curve, r_points, mse, amplitude)
        """
        # Filter NaN values
        mask = ~np.isnan(A_timeseries) & ~np.isnan(r_timeseries)
        A_clean = A_timeseries[mask]
        r_clean = r_timeseries[mask]
        
        if len(A_clean) < 2:
            raise ValueError("Not enough valid data points for fitting")
            
        min_error = np.inf
        best_min = -1
        best_fit_bessel = None
        
        # Try different minimum values to find best fit
        for i in range(len(A_clean)):
            current_min = np.min(A_clean[i:])
            
            # Calculate Bessel function with current amplitude
            lambda1 = jn_zeros(0, 1)[0]
            A_bessel = -j0(lambda1 * r_clean) * (-current_min)
            
            # Calculate error
            error = mean_squared_error(A_clean, A_bessel)
            
            # Update if better fit found
            if error < min_error:
                min_error = error
                best_fit_bessel = A_bessel
                best_min = current_min
        
        # Generate smooth curve for final plot
        R_res = 1000
        r_points = np.concatenate([-np.flip(np.arange(R_res) / R_res), 
                                  np.arange(R_res) / R_res])
        best_fit_curve = -j0(lambda1 * r_points) * (-best_min)
        
        return best_fit_curve, r_points, min_error, best_min
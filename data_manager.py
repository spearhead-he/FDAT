# data_manager.py

import numpy as np
from datetime import datetime, timedelta
import os
import sys
import re
import logging
import pandas as pd
import math

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):

        # Basic attributes
        self.satellite = None
        self.gcr_source = None
        self.start_date = None
        self.end_date = None
        self.observer_name = None
        self.output_directory = "OUTPUT"
        self.fit_type = None
        
        # Initialize coordinates
        self.distance = 1.0  # Default to 1 AU
        self.longitude = 0.0 # Default to 0 degrees
        self.latitude = 0.0  # Default to 0 degrees
        
        # Define GCR source mapping
        self.gcr_mapping = {
            'ACE': 'EPHIN',
            'WIND': 'EPHIN',
            'Helios1': 'Helios1',
            'Helios2': 'Helios2',
            'OMNI': 'EPHIN_for_OMNI',
            'SolO': 'SolO'
        }
    
        # Define detectors with complete mapping
        self.detector = {
            'ACE': 'EPHIN_F',
            'WIND': 'EPHIN_F',
            'Helios1': 'E6_D5',
            'Helios2': 'E6_D5',
            'OMNI': 'EPHIN_F',
            'SolO': 'HET'
        }

    def set_initial_params(self, satellite, start_date, end_date, observer_name):
        """Set initial parameters"""
        try:
            if satellite not in self.gcr_mapping:
                raise ValueError(f"Invalid satellite: {satellite}")
                
            self.satellite = satellite
            self.gcr_source = self.gcr_mapping[satellite]
            self.start_date = start_date
            self.end_date = end_date
            self.observer_name = observer_name
            
            logger.info(f"Initial parameters set for {satellite} with GCR source {self.gcr_source}")
            print((f"Satellite: {satellite}   GCR source {self.gcr_source}"))
            
        except Exception as e:
            logger.error(f"Error setting initial parameters: {str(e)}")
            raise

    def load_data(self):
        """Load and process all data"""
        try:
            year_start = self.start_date.year
            year_end = self.end_date.year
            doy_start = self.start_date.timetuple().tm_yday
            doy_end = self.end_date.timetuple().tm_yday
    
            print("DATA LOADING:")
            print(f"Date range: {self.start_date} to {self.end_date}")
            print(f"DOY range: {doy_start} to {doy_end}")
            
            # Load IP data first (minute resolution)
            ip_data = self.load_ip_data(year_start, year_end, doy_start, doy_end)
    
            # Set default coordinates
            self.distance = 1.0
            self.longitude = 0.0
            self.latitude = 0.0
    
            # Get first valid coordinates for Helios/SolO
            if self.satellite in ['Helios1', 'Helios2', 'SolO']:
                self.distance = np.nan
                # Find first valid distance and angles
                if 'R' in ip_data:
                    valid_dist = [x for x in ip_data['R'] if not np.isnan(x)]
                    if valid_dist:
                        self.distance = round(valid_dist[0], 2)
    
                if 'clong' in ip_data:
                    valid_long = [x for x in ip_data['clong'] if not np.isnan(x)]
                    if valid_long:
                        self.longitude = round(valid_long[0])
    
                if 'clat' in ip_data:
                    valid_lat = [x for x in ip_data['clat'] if not np.isnan(x)]
                    if valid_lat:
                        self.latitude = round(valid_lat[0])
    
            # Get IP data length for alignment
            ip_length = len(next(iter(ip_data.values())))
            doy_minute = np.linspace(doy_start, doy_end, ip_length)
            
            # Load GCR data with additional channels
            gcr_data = self.load_gcr_data(year_start, year_end, doy_start, doy_end)
            gcr = gcr_data.get('GCR', np.array([]))
            gcr_additional = gcr_data.get('GCR_additional', np.array([]))
    
            # Align GCR data with IP data resolution
            gcr_aligned = np.full(ip_length, np.nan)
            gcr_additional_aligned = np.full(ip_length, np.nan)
            
            if len(gcr) > 0:
                minute_indices = np.arange(0, ip_length, 60)
                minute_indices = minute_indices[:len(gcr)]
                gcr_aligned[minute_indices] = gcr[:len(minute_indices)]
    
            if len(gcr_additional) > 0:
                minute_indices = np.arange(0, ip_length, 60)
                minute_indices = minute_indices[:len(gcr_additional)]
                gcr_additional_aligned[minute_indices] = gcr_additional[:len(minute_indices)]
    
            result = {
                'time': doy_minute,
                'GCR': gcr_aligned,
                'GCR_HOURLY': gcr,
                'GCR_additional': gcr_additional_aligned,
                'GCR_additional_HOURLY': gcr_additional,
                **ip_data
            }
    
            return result
    
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
        
    def load_gcr_data(self, year_start, year_end, doy_start, doy_end):
        """Load GCR data based on satellite configuration"""
        try:
            if not self.gcr_source:
                raise ValueError("GCR source not configured")
    
            data = []
            data_add = []  
            
  
            for year in range(year_start, year_end + 1):
                filepath = os.path.join(
                    sys.path[0], 
                    f"data/GCR_DATA/{self.gcr_source}/{year}.txt"
                )
                print(f"Reading file: {filepath}")
                
                if not os.path.exists(filepath):
                    logger.warning(f"GCR data file not found: {filepath}")
                    continue
                    
                with open(filepath, "r") as f:
                    lines = f.readlines()
                
                for line in lines:
                    values = line.split()
                    if not values:  # Skip empty lines
                        continue
                        
                    try:
                        doy = float(values[0])
                        
                        if self.is_date_in_range(doy, year, year_start, year_end, 
                                               doy_start, doy_end):
                            if self.gcr_source in ['Helios1', 'Helios2']:
                                # Helios format: DOY val1 val2 val3 val4 val5
                                if len(values) >= 3:  # Make sure we have enough values
                                    data.append(float(values[1]) if values[1] != 'nan' else np.nan)
                                    data_add.append(float(values[2]) if values[2] != 'nan' else np.nan)
                            elif self.gcr_source == 'SolO':
                                # SolO format: sum all 
                                data.append(float(values[1]) + float(values[2])+ float(values[3])+ float(values[4])) 
                                data_add.append(float(values[2]))  # 1C_L
                            else:
                                # Standard EPHIN format
                                data.append(float(values[3]))
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Error parsing line in {filepath}: {line.strip()}")
                        continue
    
            result = {
                'GCR': np.array(data),
                'GCR_additional': np.array(data_add) if data_add else np.array([])
            }
                       
            return result
                
        except Exception as e:
            logger.error(f"Error loading GCR data: {str(e)}")
            return {'GCR': np.array([])}

    def load_ip_data(self, year_start, year_end, doy_start, doy_end):
        """Load and process IP data"""
        try:
            startMinute = (doy_start - 1) * 24 * 60
            endMinute = (doy_end - 1) * 24 * 60
            
            ip_data = {
                'B': [], 'B_fluct': [], 'Bx': [], 'By': [], 'Bz': [],
                'V': [], 'Beta': [], 'T': [], 'density': []
            }

            # Add coordinate arrays for Helios/SolO
            if self.satellite in ['Helios1', 'Helios2', 'SolO']:
                ip_data.update({
                    'R': [],      # Distance in AU
                    'clong': [],  # Carrington longitude 
                    'clat': []    # Carrington latitude
                })
            
            for year in range(year_start, year_end + 1):
                filepath = os.path.join(sys.path[0], f"data/IP/{self.satellite}/{year}.txt")
                if not os.path.exists(filepath):
                    continue
                    
                delimiter = '?' if self.satellite == 'OMNI' else ' '
                
                with open(filepath, 'r') as f:
                    lines = f.readlines()[startMinute:endMinute+1]
                
                for line in lines:
                    self.process_ip_line(line.split(delimiter), ip_data)
            
            return {k: np.array(v) for k, v in ip_data.items()}
            
        except Exception as e:
            logger.error(f"Error loading IP data: {str(e)}")
            return {}

    def process_ip_line(self, items, ip_data):
        """Process a single line of IP data"""
        try:
            def safe_float(item):
                cleaned = re.sub("[^0-9.eE-]", "", item)
                return float(cleaned) if cleaned else np.nan
                
            if len(items) >= 9:
                # Process standard fields
                ip_data['B'].append(safe_float(items[0]))
                ip_data['B_fluct'].append(safe_float(items[1]))
                ip_data['Bx'].append(safe_float(items[2]))
                ip_data['By'].append(safe_float(items[3]))
                ip_data['Bz'].append(safe_float(items[4]))
                ip_data['V'].append(safe_float(items[5]))
                
                beta = safe_float(items[6])
                if not np.isnan(beta):
                    beta = np.nan if beta > 100 else math.log2(beta)
                ip_data['Beta'].append(beta)
                
                ip_data['T'].append(safe_float(items[7]))
                ip_data['density'].append(safe_float(items[8]))

                # Process coordinate fields for Helios/SolO
                if self.satellite in ['Helios1', 'Helios2', 'SolO'] and len(items) >= 13:
                    ip_data['R'].append(safe_float(items[9]))      # Distance
                    ip_data['clat'].append(safe_float(items[11]))  # Latitude
                    ip_data['clong'].append(safe_float(items[12])) # Longitude
                
        except Exception as e:
            logger.error(f"Error processing IP line: {str(e)}")
            # Add NaN values if processing fails
            for key in ip_data:
                ip_data[key].append(np.nan)

    def is_date_in_range(self, doy, year, year_start, year_end, doy_start, doy_end):
        """Check if a date falls within the selected range"""
        current_date = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
        if year_start != year_end:
            return ((year == year_start and current_date >= self.start_date) or
                   (year == year_end and current_date <= self.end_date))
        return self.start_date <= current_date <= self.end_date

    def validate_data(self, data):
        """Validate data arrays and ensure consistent lengths within each resolution group"""
        if not data:
            return False
                
        try:
            # Get reference lengths
            minute_len = len(data.get('time', []))

            
            # Check minute resolution data
            minute_keys = ['B', 'B_fluct', 'Bx', 'By', 'Bz', 'V', 'Beta', 'T', 'density', 'time', 'GCR']
            for key in minute_keys:
                if key in data:
                    length = len(data[key])
                    if length != minute_len:
                        logger.error(f"Length mismatch for minute data {key}: {length} != {minute_len}")
                        return False

            return True
                
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False
        
    def get_data(self, key, hourly=False):
        """
        Get data array by key with option for hourly data
        Args:
            key: Data key to retrieve
            hourly: If True, return original hourly values for GCR data
        """
        try:
            data = self.load_data()
            if data is None:
                raise ValueError("No data loaded")
                
            if hourly and key == 'GCR':
                return data.get('GCR_HOURLY')
                
            if key not in data:
                raise KeyError(f"Data key '{key}' not found")
                
            return data[key]
            
        except Exception as e:
            logger.error(f"Error getting data for key {key}: {str(e)}")
            return None

    def create_output_directory(self, fit_type=None, use_date=None):
        """Create and return path for specific fit type"""
        try:
            fit_type = fit_type or self.fit_type or 'test'
            
            # Use provided date or default to start_date
            date_to_use = use_date if use_date is not None else self.start_date
            
            # Create directory name with underscores
            date_str = date_to_use.strftime('%Y_%m_%d')
            
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.output_directory,
                self.satellite,  # Put everything under satellite folder
                date_str  # Use underscore format
            )
            
            fit_dir = os.path.join(base_dir, fit_type.lower())
            logger.debug(f"Created fit directory: {fit_dir}")
            return fit_dir
                
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            raise

    def update_dates(self, new_start, new_end):
        """Update date range"""
        self.start_date = new_start
        self.end_date = new_end


# output_handler.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OutputHandler:
    def __init__(self, results_directory, script_directory):
        self.results_directory = results_directory
        self.script_directory = script_directory

    def save_parameters(self, calc_results):
        """Save calculation parameters to text files with analysis type awareness"""
        try:
            # Determine the analysis type
            is_sheath_analysis = 'region' in calc_results
            is_forbmod = 'fit' in calc_results and 'FD_bestfit' in calc_results['fit']
            
            # Save insitu results in txt file
            insitu_results = {
                'DOY Start': [calc_results['timestamps']['doy_start']],
                'DOY End': [calc_results['timestamps']['doy_end']],
                
                'vLead': [calc_results['velocities'].get('vLead', np.nan)],
                'vCenter': [calc_results['velocities'].get('v_center', np.nan)],
                'vTrail': [calc_results['velocities'].get('vTrail', np.nan)],
                'vAvg': [calc_results['velocities'].get('vAvg', np.nan)],
                'vMedian': [calc_results['velocities'].get('vMedian', np.nan)],
                'vStdev': [calc_results['velocities'].get('vStdev', np.nan)],
                'vPeak': [calc_results['velocities'].get('vPeak', np.nan)],
                
                'BPeak': [calc_results['magnetic'].get('BPeak', np.nan)],
                'BAvg': [calc_results['magnetic'].get('BAvg', np.nan)],
                'BMedian': [calc_results['magnetic'].get('BMedian', np.nan)],
                'BStdev': [calc_results['magnetic'].get('BStdev', np.nan)],
    
                'distance': [calc_results['coordinates'].get('distance', np.nan)],
                'clong': [calc_results['coordinates'].get('longitude', np.nan)],
                'clat': [calc_results['coordinates'].get('latitude', np.nan)]
            }
            
            # Add FD_obs if available for any analysis type
            if 'fd' in calc_results and 'FD_obs' in calc_results['fd']:
                insitu_results['FD_obs'] = [calc_results['fd']['FD_obs']]
            
            # Add region name for sheath analysis
            if is_sheath_analysis:
                insitu_results['region'] = [calc_results['region']]
            
            with open(os.path.join(self.results_directory, 'insitu_results.txt'), 'w') as f:
                # Write headers
                f.write(','.join(insitu_results.keys()) + '\n')
                # Write values
                f.write(','.join(str(val[0]) for val in insitu_results.values()))
    
            # Only save bestfit results if it's ForbMod and GCR data is available
            if is_forbmod and calc_results.get('has_gcr_data', False):
                bestfit_results = {
                    'FD_bestfit': [calc_results['fit']['FD_bestfit']],
                    'MSE': [f"{calc_results['fit']['MSE']:.10e}"]
                }
                
                with open(os.path.join(self.results_directory, 'bestfit_results.txt'), 'w') as f:
                    f.write(','.join(bestfit_results.keys()) + '\n')
                    f.write(','.join(str(val[0]) for val in bestfit_results.values()))
            
                # Save bestfit data
                bestfit_data = np.column_stack((
                    calc_results['fit']['r_timeseries'],
                    calc_results['fit']['A_timeseries']
                ))
                np.savetxt(
                    os.path.join(self.results_directory, 'bestfit_data.txt'),
                    bestfit_data,
                    header='r,A',
                    delimiter=','
                )
        
            logger.info(f"Parameters saved to {self.results_directory}")
                
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            raise
    
    def update_results_csv(self, sat, detector, observer, calc_results, day, fit_type, fit_categories=None, notes=""):
        """Update or create the results CSV file"""
        try:
            # Check if this is sheath analysis by looking for region field
            is_sheath_analysis = 'region' in calc_results
            region_name = calc_results.get('region', '')
            
            # Determine analysis type
            analysis_type = calc_results.get('analysis_type', 'ForbMod')
            
            # Determine CSV path based on analysis type
            if "analysis" in analysis_type.lower():
                # Already contains "analysis", don't append it again
                folder_name = analysis_type.lower().replace(' ', '_').replace('-', '')
            else:
                # Add "_analysis" suffix
                folder_name = f"{analysis_type.lower().replace(' ', '_').replace('-', '')}_analysis"
            
            sat_dir = os.path.join(self.script_directory, 'OUTPUT', sat, folder_name)
            csv_file = os.path.join(sat_dir, f"{analysis_type.lower().replace(' ', '_')}_results_{sat}_v8.csv")
            os.makedirs(sat_dir, exist_ok=True)
            
            # Get detector name and process event date
            actual_detector = detector.get(sat, "Unknown")
            year = int(day.split('/')[0])
            doy_start = calc_results['timestamps']['doy_start']
            doy_date = datetime(year, 1, 1) + timedelta(days=doy_start - 1)
            event_date = doy_date.strftime('%Y_%m_%d')  # Using _ for Excel compatibility
            
            # Create unique ID based on analysis type
            if is_sheath_analysis or analysis_type == "Sheath analysis":
                unique_id = f"Sheath_{sat}_{observer}_{doy_date.strftime('%Y_%m_%d')}_{region_name}"
            elif analysis_type == "In-situ analysis":
                unique_id = f"Insitu_{sat}_{observer}_{doy_date.strftime('%Y_%m_%d')}_{fit_type}"
            elif analysis_type == "ForbMod":
                unique_id = f"FD_{sat}_{actual_detector}_{observer}_{doy_date.strftime('%Y_%m_%d')}_{fit_type}"
            else:
                unique_id = f"Event_{sat}_{doy_date.strftime('%Y_%m_%d')}"
            
            # Load calculations from config
            config = self.load_analysis_config()
            calculations = []
            if 'analysis_types' in config and analysis_type in config['analysis_types']:
                calculations = config['analysis_types'][analysis_type].get('calculations', [])
            
            # Define standard columns that come before calculation results
            standard_columns = [
                'id', 'year', 'date', 'sat', 'detector', 
                'dist [AU]', 'clong', 'clat',
                # Add the region/borders field
                'region' if (is_sheath_analysis or analysis_type == "Sheath analysis") else 'borders',
                'DOY Start', 'DOY End'
            ]
            
            # Build ordered list of result keys from configuration to maintain order
            ordered_result_keys = []
            for calc in calculations:
                if 'result_key' in calc:
                    result_key = calc['result_key']
                    if result_key not in ordered_result_keys:  # Avoid duplicates
                        ordered_result_keys.append(result_key)
                        
            # Add ForbMod-specific columns at the end
            forbmod_columns = []
            if analysis_type == "ForbMod":
                forbmod_columns = ['FD_bestfit', 'MSE']
                
            # Final columns for notes and fit type
            final_columns = ['fit type', 'notes']
            
            # Read existing data
            rows = []
            header = None
            if os.path.isfile(csv_file):
                with open(csv_file, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    header = reader.fieldnames
                    rows = list(reader)
            
            # Define our preferred header order
            preferred_header = []
            for col in standard_columns:
                preferred_header.append('region' if col == 'region' and is_sheath_analysis else 
                                       'borders' if col == 'borders' and not is_sheath_analysis else col)
            preferred_header.extend(ordered_result_keys)
            preferred_header.extend(forbmod_columns)
            preferred_header.extend(final_columns)
            
            # Build results row with all possible fields from preferred header
            results_row = {col: "" for col in preferred_header}
            
            # Fill in standard values
            results_row.update({
                'id': unique_id,
                'year': day.split('/')[0],
                'date': event_date,
                'sat': sat,
                'detector': actual_detector,
                'dist [AU]': self._format_value(calc_results.get('coordinates', {}).get('distance'), "{:.2f}"),
                'clong': self._format_value(calc_results.get('coordinates', {}).get('longitude'), "{:.1f}"),
                'clat': self._format_value(calc_results.get('coordinates', {}).get('latitude'), "{:.1f}")
            })
            
            # Add borders/region field
            if is_sheath_analysis or analysis_type == "Sheath analysis":
                results_row['region'] = region_name
            else:
                results_row['borders'] = fit_type
            
            # Add DOY values
            results_row['DOY Start'] = self._format_value(calc_results.get('timestamps', {}).get('doy_start'), "{:.2f}")         
            results_row['DOY End'] = self._format_value(calc_results.get('timestamps', {}).get('doy_end'), "{:.2f}")
            
            # Add calculation results
            for calc in calculations:
                if 'result_key' in calc:
                    result_key = calc['result_key']
                    value = None
                    
                    # Find value in calc_results (check all sections)
                    for section in ['parameters', 'magnetic', 'velocities', 'plasma', 'fd']:
                        if section in calc_results and result_key in calc_results[section]:
                            value = calc_results[section][result_key]
                            break
                    
                    # Add to results row if found
                    if value is not None:
                        results_row[result_key] = self._format_value(value, "{:.2f}")
            
            # Add FD_bestfit and MSE for ForbMod
            if analysis_type == "ForbMod":
                results_row['FD_bestfit'] = self._format_value(calc_results.get('fit', {}).get('FD_bestfit'), "{:.2f}")
                results_row['MSE'] = self._format_value(calc_results.get('fit', {}).get('MSE'), "{:.6e}")
            
            # Add fit type and notes
            if fit_categories:
                results_row['fit type'] = ", ".join(fit_categories)
            results_row['notes'] = notes
            
            # If no header exists, use our preferred order
            if not header:
                header = preferred_header
            else:
                # Make sure any new columns are added to the header
                for col in preferred_header:
                    if col not in header:
                        header.append(col)
            
            # Update existing row or add new one
            row_found = False
            for row in rows:
                match_condition = (row.get('date') == event_date and 
                                  (row.get('region' if is_sheath_analysis else 'borders') == 
                                   (region_name if is_sheath_analysis else fit_type)))
                if match_condition:
                    row.update(results_row)
                    row_found = True
                    break
                    
            if not row_found:
                rows.append(results_row)
            
            # Write to CSV with consistent header order
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Results CSV updated: {csv_file}")
            
        except Exception as e:
            logger.error(f"Error updating results CSV: {str(e)}", exc_info=True)
            raise
    
    def _format_value(self, value, format_str):
        """Helper method to format values with proper error handling"""
        if value is None:
            return ""
        try:
            return format_str.format(float(value))
        except (ValueError, TypeError):
            return str(value)

    def save_plot(self, fig, filename='best_fit.jpg'):
        """Save the plot figure"""
        try:
            filepath = os.path.join(self.results_directory, filename)
            fig.savefig(filepath, dpi=100)
            logger.info(f"Plot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            raise
    
    def save_all_plots(self, fig, calc_results):
        """Save all plot versions and results"""
        try:
            if not calc_results:
                logger.warning("No calculation results to save")
                return      
            
            # Save params to txt files
            self.save_parameters(calc_results)
                
            logger.info(f"All plots and data saved to {self.results_directory}")
            
        except Exception as e:
            logger.error(f"Error saving plots and data: {str(e)}")
            raise

    def load_analysis_config(self):
        """Load analysis configuration from JSON file"""
        try:
            import json
            
            config_path = os.path.join(self.script_directory, 'analysis-config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config
            else:
                logger.warning(f"Analysis config file not found: {config_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading analysis config: {str(e)}")
            return {}

    def save_publication_figure(self, fig, satellite, event_date, analysis_type, subtype=None):
        """Save publication-quality figure with descriptive filename
        
        Args:
            fig: Matplotlib figure to save
            satellite: Satellite name
            event_date: Event date (datetime object)
            analysis_type: Type of analysis ('ForbMod', 'In-situ analysis', 'Sheath analysis')
            subtype: Optional subtype (fit type or region name)
        
        Returns:
            Path to saved figure
        """
        try:
            # Format date as YYYYMMDD
            date_str = event_date.strftime('%Y%m%d')
            
            # Format satellite name (handle special case writing for neutron monitors)
            sat_name = 'nm' if satellite == 'neutron monitors' else satellite
            
            # Create base filename
            if analysis_type == "ForbMod":
                analysis_abbr = "forbmod"
            elif analysis_type == "In-situ analysis":
                analysis_abbr = "insitu"
            else: 
                analysis_abbr = "sheath"
                
            # Add subtype if provided
            if subtype:
                filename = f"{sat_name}_{date_str}_{analysis_abbr}_{subtype}.png"
            else:
                filename = f"{sat_name}_{date_str}_{analysis_abbr}.png"
                
            # Save figure with descriptive name
            filepath = os.path.join(self.results_directory, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            
            logger.info(f"Publication figure saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving publication figure: {str(e)}")
            return None


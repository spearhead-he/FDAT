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
        
    def save_plot(self, fig, filename='best_fit.jpg'):
        """Save the plot figure"""
        try:
            filepath = os.path.join(self.results_directory, filename)
            fig.savefig(filepath)
            logger.info(f"Plot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            raise

    def save_parameters(self, params, calc_results):
        """Save calculation parameters to text files"""
        try:
            # Save insitu results as column format
            insitu_results = {
                'DOY Start': [calc_results['timestamps']['doy_start']],
                'DOY End': [calc_results['timestamps']['doy_end']],
                'vLead': [calc_results['velocities']['vLead']],
                'vTrail': [calc_results['velocities']['vTrail']],
                'BPeak': [calc_results['magnetic']['BPeak']],
                'BAvg': [calc_results['magnetic']['BAvg']],
                'FD_obs': [calc_results['fd']['FD_obs']]
            }
            
            with open(os.path.join(self.results_directory, 'insitu_results.txt'), 'w') as f:
                # Write headers
                f.write(','.join(insitu_results.keys()) + '\n')
                # Write values
                f.write(','.join(str(val[0]) for val in insitu_results.values()))


            # Only save bestfit results if GCR data is available
            if calc_results.get('has_gcr_data', False):
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
    
    def update_results_csv(self, sat, detector, observer, calc_results, day, fit_type, fit_categories):
        """Update or create the results CSV file"""
        try:
            # Put CSV file in satellite folder
            sat_dir = os.path.join(self.script_directory, 'OUTPUT', sat)
            os.makedirs(sat_dir, exist_ok=True)
            csv_file = os.path.join(sat_dir, f"all_res_{sat}.csv")
            
            # Define header here
            header = ['id', 'year', 'date','sat', 'detector', 'dist [AU]', 'clong', 'clat',
                     'borders', 'DOY Start', 'DOY End', 'DOY FDmin',
                     'vAvg [km/s]', 'vMedian', 'vStdev', 'vLead', 'v_center',
                     'vTrail', 'upstream_w', 'BPeak [nT]', 'BAvg', 'BMedian',
                     'BStdev', 'FD_obs [%]', 'FD_bestfit', 'MSE', 'fit type']
            
            # Get actual detector name instead of full dictionary
            actual_detector = detector.get(sat, "Unknown")

            # Get year from the original day string
            year = int(day.split('/')[0])
            
            # Convert doy_start to date
            doy_date = datetime(year, 1, 1) + timedelta(days=calc_results['timestamps']['doy_start'] - 1)
            event_date = doy_date.strftime('%Y/%m/%d')
            
            # Create unique ID including observer
            unique_id = f"FD_{sat}_{actual_detector}_{observer}_{doy_date.strftime('%Y_%m_%d')}_{fit_type}"
            
            # Create results row
            results_row = {
                'id': unique_id,
                'year': day.split('/')[0],
                'date': event_date,
                'sat': sat,
                'detector': actual_detector,
                'dist [AU]': calc_results['coordinates']['distance'],
                'clong': calc_results['coordinates']['longitude'],
                'clat': calc_results['coordinates']['latitude'],
                'borders': fit_type,
                'DOY Start': calc_results['timestamps']['doy_start'],
                'DOY End': calc_results['timestamps']['doy_end'],
                'DOY FDmin': calc_results['timestamps']['FD_min_DOY'],
                'vAvg [km/s]': calc_results['velocities']['vAvg'],
                'vMedian': calc_results['velocities']['vMedian'],
                'vStdev': calc_results['velocities']['vStdev'],
                'vLead': calc_results['velocities']['vLead'],
                'v_center': calc_results['velocities']['v_center'],
                'vTrail': calc_results['velocities']['vTrail'],
                'upstream_w': calc_results['velocities']['upstream_w'],
                'BPeak [nT]': calc_results['magnetic']['BPeak'],
                'BAvg': calc_results['magnetic']['BAvg'],
                'BMedian': calc_results['magnetic']['BMedian'],
                'BStdev': calc_results['magnetic']['BStdev'],

                'FD_obs [%]': calc_results['fd']['FD_obs'] if calc_results.get('has_gcr_data', False) else np.nan,
                'FD_bestfit': calc_results['fit']['FD_bestfit'] if calc_results.get('has_gcr_data', False) else np.nan,
                'MSE': calc_results['fit']['MSE'] if calc_results.get('has_gcr_data', False) else np.nan,
            }
    
            # Read existing data
            rows = []
            if os.path.isfile(csv_file):
                with open(csv_file, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
    
            # Update or add row
            row_found = False
            for row in rows:
                if (row['date'] == day and row['borders'] == fit_type):
                    row.update(results_row)
                    row_found = True
                    break
    
            if not row_found:
                rows.append(results_row)
    
            # Write updated data
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
    
            logger.info(f"Results CSV updated: {csv_file}")
        except Exception as e:
            logger.error(f"Error updating results CSV: {str(e)}")
            raise
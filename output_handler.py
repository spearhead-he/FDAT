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
        """Update or create the results CSV file with concatenated fit categories"""
        try:
            # Put CSV file in satellite folder
            sat_dir = os.path.join(self.script_directory, 'OUTPUT', sat)
            os.makedirs(sat_dir, exist_ok=True)
            csv_file = os.path.join(sat_dir, f"all_res_{sat}.csv")
            
            # Define header (without separate category columns)
            header = ['id', 'year', 'date', 'sat', 'detector', 'dist [AU]', #'clong', 'clat',
                     'borders', 'DOY Start', 'DOY End', 'DOY FDmin',
                     'vAvg [km/s]', 'vMedian', 'vStdev', 'vLead', 'v_center',
                     'vTrail', 'upstream_w', 'BPeak [nT]', 'BAvg', 'BMedian',
                     'BStdev', 'FD_obs [%]', 'FD_bestfit', 'MSE', 'fit type']
            
            # Get actual detector name
            actual_detector = detector.get(sat, "Unknown")
            
            # Get year from the day string
            year = int(day.split('/')[0])
            
            # Convert doy_start to date
            doy_date = datetime(year, 1, 1) + timedelta(days=calc_results['timestamps']['doy_start'] - 1)
            event_date = doy_date.strftime('%Y/%m/%d')
            
            # Create unique ID
            unique_id = f"FD_{sat}_{actual_detector}_{observer}_{doy_date.strftime('%Y_%m_%d')}_{fit_type}"
            
            # Combine fit categories into a single string
            fit_description = ", ".join(fit_categories) if fit_categories else ""
            
            # Create results row
            results_row = {
                'id': unique_id,
                'year': day.split('/')[0],
                'date': event_date,
                'sat': sat,
                'detector': actual_detector,
                'dist [AU]': calc_results['coordinates']['distance'],
                #'clong': calc_results['coordinates']['longitude'],
                #'clat': calc_results['coordinates']['latitude'],
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
                'fit type': fit_description
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

    def save_publication_plot(self, fig, filename='publication_plot.png'):
            """Save the publication-quality matplotlib figure"""
            try:
                # Ensure the directory exists
                os.makedirs(self.results_directory, exist_ok=True)
                
                # Define filepath
                filepath = os.path.join(self.results_directory, filename)
                
                # Save with high quality settings
                fig.savefig(filepath, 
                           dpi=300,                    # High resolution
                           bbox_inches='tight',        # Trim whitespace
                           pad_inches=0.1,             # Small padding
                           facecolor='white',          # White background
                           edgecolor='none',           # No edge color
                           format='png',               # PNG format
                           transparent=False)          # Solid background
                
                # Also save as PDF for vector graphics
                pdf_filepath = os.path.splitext(filepath)[0] + '.pdf'
                fig.savefig(pdf_filepath,
                           format='pdf',
                           bbox_inches='tight',
                           pad_inches=0.1)
                
                logger.info(f"Publication plot saved to {filepath} and {pdf_filepath}")
                
            except Exception as e:
                logger.error(f"Error saving publication plot: {str(e)}")
                raise

    def save_parameters(self, params=None, calc_results=None, output_info=None):
        """Save calculation parameters to text files"""
        try:
            # Fix parameter handling
            if calc_results is None:
                calc_results = params  # Use the first argument as calc_results
    
            if not calc_results:
                logger.warning("No calculation results to save")
                return
                
                
            # Save insitu results
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
                f.write(','.join(insitu_results.keys()) + '\n')
                f.write(','.join(str(val[0]) for val in insitu_results.values()))
    
            # Save bestfit results if available
            if 'fit' in calc_results:
                bestfit_results = {
                    'FD_bestfit': [calc_results['fit'].get('FD_bestfit', 0)],
                    'MSE': [calc_results['fit'].get('MSE', 0)]
                }
                
                with open(os.path.join(self.results_directory, 'bestfit_results.txt'), 'w') as f:
                    f.write(','.join(bestfit_results.keys()) + '\n')
                    f.write(f"{bestfit_results['FD_bestfit'][0]},{bestfit_results['MSE'][0]}")
            
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            raise

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
                
            # Save plot window
            filepath = os.path.join(self.results_directory, 'plot_window.png')
            fig.savefig(filepath, dpi=100)
            
            # Save parameters
            self.save_parameters(calc_results)
            
            # Save fit data if available
            fit_data = calc_results.get('fit', {})
            if fit_data:
                r_timeseries = fit_data.get('r_timeseries')
                a_timeseries = fit_data.get('A_timeseries')
                
                if r_timeseries is not None and a_timeseries is not None:
                    bestfit_data = np.column_stack((r_timeseries, a_timeseries))
                    np.savetxt(
                        os.path.join(self.results_directory, 'bestfit_data.txt'),
                        bestfit_data,
                        header='r,A',
                        delimiter=','
                    )
                
            logger.info(f"All plots and data saved to {self.results_directory}")
            
        except Exception as e:
            logger.error(f"Error saving plots and data: {str(e)}")
            raise
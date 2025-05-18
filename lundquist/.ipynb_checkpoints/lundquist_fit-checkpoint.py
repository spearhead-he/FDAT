# lundquist_fit.py - Combined Lundquist fitting functions

import numpy as np
import datetime
from scipy.special import jn_zeros, j0, j1
import os
import lmfit
import matplotlib.pyplot as plt
import logging
import traceback

logger = logging.getLogger(__name__)

# ----- DATA PREPARATION FUNCTIONS -----

def load_from_gui_data(data_dict, region_datetime):
    """Convert GUI data to format needed by fitting code
    
    Args:
        data_dict: Dictionary with MF and SW data from the GUI
        region_datetime: Dictionary with region timestamps
        
    Returns:
        time, btot, br, bt, bn, vel arrays suitable for fitting
    """
    try:
        # Extract the time range for MO region
        start_dt = region_datetime['mo_start']
        end_dt = region_datetime['mo_end']
        
        # Extract MF data
        mf_data = data_dict.get('mf', {})
        if not mf_data or 'time' not in mf_data:
            raise ValueError("Missing magnetic field data")
            
        # Find indices within the region
        time_np = np.array(mf_data['time'])
        mask = (time_np >= np.datetime64(start_dt)) & (time_np <= np.datetime64(end_dt))
        
        # Extract time and convert to hours from start
        time_filtered = time_np[mask]
        if len(time_filtered) == 0:
            raise ValueError("No data points found in MO region")
            
        # Convert times to hours from start
        first_time = time_filtered[0]
        time = np.array([(t - first_time).astype('timedelta64[s]').astype(float)/3600 for t in time_filtered])
        
        # Extract B components (RTN)
        if 'Br' in mf_data:
            br = mf_data['Br'][mask]
        elif 'Bx' in mf_data:
            br = mf_data['Bx'][mask]
        else:
            raise ValueError("Missing radial magnetic field component (Br/Bx)")
            
        if 'Bt' in mf_data:
            bt = mf_data['Bt'][mask]
        elif 'By' in mf_data:
            bt = mf_data['By'][mask]
        else:
            raise ValueError("Missing tangential magnetic field component (Bt/By)")
            
        if 'Bn' in mf_data:
            bn = mf_data['Bn'][mask]
        elif 'Bz' in mf_data:
            bn = mf_data['Bz'][mask]
        else:
            raise ValueError("Missing normal magnetic field component (Bn/Bz)")
        
        # Calculate total B
        btot = np.sqrt(br**2 + bt**2 + bn**2)
        
        # Extract velocity from SW data
        sw_data = data_dict.get('sw', {})
        if sw_data and 'time' in sw_data and 'V' in sw_data:
            sw_time_np = np.array(sw_data['time'])
            sw_mask = (sw_time_np >= np.datetime64(start_dt)) & (sw_time_np <= np.datetime64(end_dt))
            vel = sw_data['V'][sw_mask]
            
            # Ensure vel array is same length as magnetic field data
            if len(vel) != len(time):
                # Resample velocity to match magnetic field data time points
                vel_resampled = np.full(len(time), np.nan)
                sw_time_hours = np.array([(t - first_time).astype('timedelta64[s]').astype(float)/3600 for t in sw_time_np[sw_mask]])
                
                for i, t in enumerate(time):
                    # Find closest time point
                    if len(sw_time_hours) > 0:
                        closest_idx = np.argmin(np.abs(sw_time_hours - t))
                        if abs(sw_time_hours[closest_idx] - t) < 0.1:  # Within 6 minutes
                            vel_resampled[i] = sw_data['V'][sw_mask][closest_idx]
                
                vel = vel_resampled
        else:
            # Use NaN for velocity if not available
            vel = np.full(len(time), np.nan)
        
        # Extract density and temperature if available
        den = np.full(len(time), np.nan)
        tem = np.full(len(time), np.nan)
        
        if sw_data and 'N' in sw_data:
            sw_time_np = np.array(sw_data['time'])
            sw_mask = (sw_time_np >= np.datetime64(start_dt)) & (sw_time_np <= np.datetime64(end_dt))
            if len(sw_mask) > 0 and np.any(sw_mask):
                sw_time_hours = np.array([(t - first_time).astype('timedelta64[s]').astype(float)/3600 for t in sw_time_np[sw_mask]])
                for i, t in enumerate(time):
                    if len(sw_time_hours) > 0:
                        closest_idx = np.argmin(np.abs(sw_time_hours - t))
                        if abs(sw_time_hours[closest_idx] - t) < 0.1:  # Within 6 minutes
                            den[i] = sw_data['N'][sw_mask][closest_idx]
                
        if sw_data and 'T' in sw_data:
            sw_time_np = np.array(sw_data['time'])
            sw_mask = (sw_time_np >= np.datetime64(start_dt)) & (sw_time_np <= np.datetime64(end_dt))
            if len(sw_mask) > 0 and np.any(sw_mask):
                sw_time_hours = np.array([(t - first_time).astype('timedelta64[s]').astype(float)/3600 for t in sw_time_np[sw_mask]])
                for i, t in enumerate(time):
                    if len(sw_time_hours) > 0:
                        closest_idx = np.argmin(np.abs(sw_time_hours - t))
                        if abs(sw_time_hours[closest_idx] - t) < 0.1:  # Within 6 minutes
                            tem[i] = sw_data['T'][sw_mask][closest_idx]
        
        return time, btot, br, bt, bn, vel, den, tem, first_time
    
    except Exception as e:
        logger.error(f"Error loading data from GUI: {str(e)}")
        raise

# From prep_data.py
def extract(rope_bounds, time, br, bt, bn, vel):
    """Extract the data within the rope"""
    
    mask = np.where(np.logical_and(time >= rope_bounds[0], time <= rope_bounds[1]))
    
    t_rope = time[mask[0][0]:mask[-1][-1]+1]
    br_rope = br[mask[0][0]:mask[-1][-1]+1]
    bt_rope = bt[mask[0][0]:mask[-1][-1]+1]
    bn_rope = bn[mask[0][0]:mask[-1][-1]+1]
    vel_rope = vel[mask[0][0]:mask[-1][-1]+1]
    
    return t_rope, br_rope, bt_rope, bn_rope, vel_rope

def get_vel(vel_rope, t_rope):
    """Get the average speed"""
    
    vel_avg = np.nanmean(vel_rope)    
    if np.isnan(vel_avg):
        vel_avg = 400
    
    return vel_avg

# From rope_fit.py
def rotate_coords(theta0, phi0):
    """Define the coords of cloud frame"""
    # Axis is basically zc_hat!
    
    r_hat = [-1, 0, 0]
    
    theta0 = (np.pi/2) - np.deg2rad(theta0)
    phi0 = np.deg2rad(phi0)
    
    axis = [ np.sin(theta0)*np.cos(phi0), np.sin(theta0)*np.sin(phi0), np.cos(theta0) ]
    axis /= np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)

    yc_hat = np.cross(axis, r_hat)
    yc_hat /= np.sqrt(yc_hat[0]**2 + yc_hat[1]**2 + yc_hat[2]**2)
        
    xc_hat = np.cross(yc_hat, axis)
    
    return xc_hat, yc_hat, axis

def transform_cloud_frame(xc_hat, yc_hat, axis, br_rope, bt_rope, bn_rope):
    """Transform data into cloud frame"""
    
    r_hat = [1, 0, 0]
    t_hat = [0, 1, 0]
    n_hat = [0, 0, 1]
    
    br_cloud = br_rope*np.dot(r_hat, xc_hat) + bt_rope*np.dot(t_hat, xc_hat) + bn_rope*np.dot(n_hat, xc_hat)
    bt_cloud = br_rope*np.dot(r_hat, yc_hat) + bt_rope*np.dot(t_hat, yc_hat) + bn_rope*np.dot(n_hat, yc_hat)
    bn_cloud = br_rope*np.dot(r_hat, axis) + bt_rope*np.dot(t_hat, axis) + bn_rope*np.dot(n_hat, axis)
    
    return br_cloud, bt_cloud, bn_cloud

def get_sampling_points(t_rope, p0, r0, t0):
    """Derive the sampling points within the rope as collection of radii"""
    
    r_in_time = r0 * (1 + (t_rope/t0))
    exp_fac = (1 + (t_rope / t0))
    
    xmax = np.sqrt(1-(p0**2))    
    x_cross = np.linspace(-xmax,xmax,len(t_rope)) / exp_fac
    x_cross_revamped = ( ( (x_cross - np.nanmin(x_cross)) / (np.nanmax(x_cross) - np.nanmin(x_cross)) ) * 2 * xmax ) - xmax
    
    y_cross = p0 / exp_fac
    
    samples = np.sqrt( x_cross_revamped**2 + y_cross**2 )
    
    return x_cross_revamped, samples

def get_field_components(t_rope, t0, b0, h, samples):
    """Cylinder model in the cloud frame"""
    
    alpha = 2.405
    
    exp_fac = (1 + (t_rope / t0))
    
    b_axial = ( (b0/(exp_fac**2))  * j0((alpha*samples)/exp_fac) )
    b_tangent = ( (b0/(exp_fac**2)) * h * j1((alpha*samples)/exp_fac) )
    
    return b_axial, b_tangent

def cloud_to_cartesian(b_axial, b_tangent, xc_hat, yc_hat, axis, p0, x_cross, samples):
    """Reproject cylinder coordinates into RTN"""
    
    angle = np.arcsin( x_cross / samples )
    
    if (p0 > 0):
        cosine = -np.cos(angle)
    else:
        cosine = np.cos(angle)
    
    bx_cloud = b_tangent * cosine
    by_cloud = b_tangent * np.sin(angle)
    bz_cloud = b_axial
    
    r_hat = [1, 0, 0]
    t_hat = [0, 1, 0]
    n_hat = [0, 0, 1]
    
    br_fit = bx_cloud*np.dot(r_hat, xc_hat) + by_cloud*np.dot(r_hat, yc_hat) + bz_cloud*np.dot(r_hat, axis)
    bt_fit = bx_cloud*np.dot(t_hat, xc_hat) + by_cloud*np.dot(t_hat, yc_hat) + bz_cloud*np.dot(t_hat, axis)
    bn_fit = bx_cloud*np.dot(n_hat, xc_hat) + by_cloud*np.dot(n_hat, yc_hat) + bz_cloud*np.dot(n_hat, axis)
    
    btot_fit = np.sqrt(br_fit**2 + bt_fit**2 + bn_fit**2)
    
    return btot_fit, br_fit, bt_fit, bn_fit

def get_modelled_speed(t_rope, samples, x_cross, vel_avg, t0, r0, p0):
    """Get the fitted Vr"""
    
    angles = np.arcsin( x_cross / samples )    
    r_in_time = r0 * (1 + (t_rope/t0))
    
    # Conversion unit constant from the original code
    # In the original code, this uses astropy.units, but we're simplifying
    v0_exp = r0/t0 * 1.496e8 / 3600  # AU/h to km/s
    
    vxc_exp = v0_exp * samples * np.sin(angles)
    vyc_exp = v0_exp * samples * np.cos(angles)
    
    vel_fit = np.sqrt( (vel_avg-vxc_exp)**2 + vyc_exp**2 )

    return vel_fit

def get_cloud_radius(t_rope, vel_avg, theta0, phi0, p0):
    """Get the cloud radius"""
        
    pass_time = (t_rope[-1] - t_rope[0]) * 3600.
    vel_proj = vel_avg * np.sqrt( np.sin(np.deg2rad(theta0))**2 + ( np.cos(np.deg2rad(theta0))*np.sin(np.deg2rad(phi0)) )**2 )
    
    r_cloud = ( pass_time * vel_proj ) / ( 2 * np.sqrt( 1 - (p0**2) ) )
    
    # Convert km to AU
    r0 = r_cloud / 1.496e8
    
    return r0

def get_rope_fluxes(b0, r0):
    """Get the rope fluxes"""
        
    # Convert nT to Gauss
    b_gauss = b0 * 1e-5
    
    # Convert AU to cm
    r_cm = r0 * 1.496e13
    
    flu_axis = 1.35643 * b_gauss * r_cm**2
    flu_polo = 0.41584 * b_gauss * r_cm
    
    # Convert flux per cm to flux per AU
    flu_polo = flu_polo * 1.496e13
    
    return flu_axis, flu_polo

def fit_initial_guess(fparam, t_rope, data):
    """The mega fit function"""
    
    par = fparam.valuesdict()
    theta0 = par['theta0']
    phi0 = par['phi0']
    p0 = par['p0']
    h = par['h']
    b0 = par['b0']
    t0 = par['t0']
    
    br_rope = data[0]
    bt_rope = data[1]
    bn_rope = data[2]
    vel_rope = data[3]
    
    # Get the rope mean speed
    vel_avg = get_vel(vel_rope, t_rope)
    
    # Get the corresponding r0
    r0 = get_cloud_radius(t_rope, vel_avg, theta0,  phi0, p0)
    
    # Get the unit vectors of the cloud frame
    xc_hat, yc_hat, axis = rotate_coords(theta0, phi0)

    # Transform the RTN data into cloud frame
    br_cloud, bt_cloud, bn_cloud = transform_cloud_frame(xc_hat, yc_hat, axis, br_rope, bt_rope, bn_rope)

    # Get the measurement points through the cloud as a set of radiuses from centre of cloud to each measurement
    x_cross, samples = get_sampling_points(t_rope, p0, r0, t0)

    # Get the fitted flux rope Bfield
    b_axial, b_tangent = get_field_components(t_rope, t0, b0, h, samples)

    # Transform back to RTN
    btot_fit, br_fit, bt_fit, bn_fit = cloud_to_cartesian(b_axial, b_tangent, xc_hat, yc_hat, axis, p0, x_cross, samples)
    
    # Get modelled speed
    vel_fit = get_modelled_speed(t_rope, samples, x_cross, vel_avg, t0, r0, p0)
    
    return btot_fit, br_fit, bt_fit, bn_fit, vel_fit

# From optimise_fit.py
def normall(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, b0):
    """Normalise all the field components (obs & model)"""
    
    btot_rope = np.sqrt( br_rope**2 + bt_rope**2 + bn_rope**2 )
    
    br_rope_norm = br_rope / np.nanmax(btot_rope)
    bt_rope_norm = bt_rope / np.nanmax(btot_rope)
    bn_rope_norm = bn_rope / np.nanmax(btot_rope)
    
    br_fit_norm = br_fit / b0
    bt_fit_norm = bt_fit / b0
    bn_fit_norm = bn_fit / b0
    
    return br_rope_norm, bt_rope_norm, bn_rope_norm, br_fit_norm, bt_fit_norm, bn_fit_norm

def get_chi_dir(br_rope_norm, bt_rope_norm, bn_rope_norm, br_fit_norm, bt_fit_norm, bn_fit_norm):
    """Define and obtain the chi2 metric for direction"""
    
    chi_dir = np.sum( (br_rope_norm-br_fit_norm)**2 + (bt_rope_norm-bt_fit_norm)**2 + (bn_rope_norm-bn_fit_norm)**2 ) / len(br_rope_norm)
    
    return chi_dir

def get_chi_mag(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, b0):
    """Define and obtain the chi2 metric for magnitude"""
    
    btot_rope = np.sqrt( br_rope**2 + bt_rope**2 + bn_rope**2 )
    btot_fit = np.sqrt( br_fit**2 + bt_fit**2 + bn_fit**2 )
    
    chi_mag = np.sum( (btot_rope-btot_fit)**2 ) / ( len(br_rope) * b0)
    
    return chi_mag

def get_chi(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, b0):
    """Calculate both CHIs"""
    
    br_rope_norm, bt_rope_norm, bn_rope_norm, br_fit_norm, bt_fit_norm, bn_fit_norm = normall(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, b0)
    
    chi_dir = get_chi_dir(br_rope_norm, bt_rope_norm, bn_rope_norm, br_fit_norm, bt_fit_norm, bn_fit_norm)
    chi_mag = get_chi_mag(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, b0)
    
    return chi_dir, chi_mag

def fit_forever(fparam, t_rope, keyword, data=None):
    """Function to minimise --- All in one go"""
    
    par = fparam.valuesdict()
    theta0 = par['theta0']
    phi0 = par['phi0']
    p0 = par['p0']
    h = par['h']
    b0 = par['b0']
    t0 = par['t0']

    br_rope = data[0]
    bt_rope = data[1]
    bn_rope = data[2]
    vel_rope = data[3]
    
    # Get the rope mean speed - avoid NaN issues
    vel_avg = get_vel(vel_rope, t_rope)
    
    # Safety check to avoid division by zero in radius calculation
    if t0 <= 0:
        t0 = 1.0  # Minimum 1 hour for expansion time
    
    # Add safeguards for calculations that might produce NaN
    try:
        # Get the corresponding r0 with protection for edge cases
        r0 = get_cloud_radius(t_rope, vel_avg, theta0, phi0, p0)
        
        # Check for invalid r0 values
        if r0 <= 0 or np.isnan(r0):
            r0 = 0.1  # Safe default radius in AU
            
        # Protect against p0 values too close to ±1 which cause math domain errors
        if abs(p0) >= 0.99:
            p0 = 0.99 * (p0 / abs(p0))  # Keep sign but limit absolute value
            
        # Get the unit vectors of the cloud frame
        xc_hat, yc_hat, axis = rotate_coords(theta0, phi0)
    
        # Transform the RTN data into cloud frame
        br_cloud, bt_cloud, bn_cloud = transform_cloud_frame(xc_hat, yc_hat, axis, br_rope, bt_rope, bn_rope)
    
        # Get the measurement points through the cloud as a set of radiuses from centre of cloud to each measurement
        x_cross, samples = get_sampling_points(t_rope, p0, r0, t0)
    
        # Get the fitted flux rope Bfield
        b_axial, b_tangent = get_field_components(t_rope, t0, b0, h, samples)
    
        # Transform back to RTN
        btot_fit, br_fit, bt_fit, bn_fit = cloud_to_cartesian(b_axial, b_tangent, xc_hat, yc_hat, axis, p0, x_cross, samples)
        
        # Get modelled speed
        vel_fit = get_modelled_speed(t_rope, samples, x_cross, vel_avg, t0, r0, p0)
        
        # Calculate residuals with protection against NaN
        res_bb = np.sqrt(np.clip(br_rope**2 + bt_rope**2 + bn_rope**2, 0, None)) - btot_fit
        res_br = br_rope - br_fit
        res_bt = bt_rope - bt_fit
        res_bn = bn_rope - bn_fit
        
        # Handle velocity data safely
        if np.all(np.isnan(vel_rope)):
            res_vv = np.array([])
        else:
            res_vv = vel_rope - vel_fit
            
    except Exception as e:
        # If any calculation errors occurred, return dummy residuals
        print(f"Warning in fit calculations: {str(e)}")
        return np.ones(len(t_rope)) * 1e6  # Return large residuals to signal bad fit
    
    # Determine final residuals based on keyword
    if keyword == 'all':        
        if len(res_vv) == 0 or np.all(np.isnan(res_vv)):
            residuals = np.concatenate([res_bb, res_br, res_bt, res_bn])
        else:
            residuals = np.concatenate([res_bb, res_br, res_bt, res_bn, res_vv[~np.isnan(res_vv)]])
            
    elif keyword == 'step1':       
        if len(res_vv) == 0 or np.all(np.isnan(res_vv)):
            residuals = np.concatenate([res_br, res_bt, res_bn])
        else:
            residuals = np.concatenate([res_br, res_bt, res_bn, res_vv[~np.isnan(res_vv)]])
            
    elif keyword == 'step2':
        residuals = res_bb
        
    else:
        print("Please clarify what to compare your fits with.")
        print("Choices: [all] - [step1] - [step2]")
        return np.ones(len(t_rope)) * 1e6
    
    # Final protection against NaN values
    if np.any(np.isnan(residuals)):
        # Replace NaNs with large values to guide optimizer away from these regions
        residuals = np.nan_to_num(residuals, nan=1e6)
    
    return residuals



# From output_fit.py
def print_params(filename, out_dir, theta0, phi0, p0, h, b0, t0, r0, t_rope, flu_axis, flu_polo, chi_dir, chi_mag):
    """Spit out the results"""
    
    if (h==1):
        hs = 'p'
    elif (h==-1):
        hs = 'n'
    
    # Calculate expansion speed in km/s
    vexp = (r0/t0) * 1.496e8 / 3600  # AU/h to km/s
    
    r_in_time = r0 * (1 + (t_rope/t0))
    
    target = open(os.path.join(out_dir, filename), 'w')
    target.write('########################################################\n')
    target.write('# EXPANDING LUNDQUIST FIT RESULTS #\n')
    target.write('########################################################\n')
    target.write('\n')
    target.write('####### Rope parameters\n')
    target.write('Axis latitude = '+str("{:.2f}".format(theta0))+' deg\n')
    target.write('Axis longitude = '+str("{:.2f}".format(phi0))+' deg\n')
    target.write('Impact parameter = '+str("{:.3f}".format(p0))+' [normalised by R0]\n')
    target.write('Helicity sign = '+str(h)+'\n')
    target.write('Axial field = '+str("{:.2f}".format(b0))+' nT\n')
    target.write('Expansion time = '+str("{:.2f}".format(t0))+' h\n')
    target.write('\n')
    target.write('####### Derived quantities\n')
    target.write('Initial rope radius (t0) = '+str("{:.3f}".format(r0))+' au\n')
    target.write('Final rope radius (end) = '+str("{:.3f}".format(r_in_time[-1]))+' au\n')
    target.write('Expansion speed = '+str("{:.3f}".format(vexp))+' km/s\n')
    target.write('Axial flux = '+str("{:.2e}".format(flu_axis))+' Mx\n')
    target.write('Poloidal flux = '+str("{:.2e}".format(flu_polo))+' Mx/au\n')
    target.write('\n')
    target.write('####### Goodness of fit\n')
    target.write('chi^2 dir = '+str("{:.3f}".format(chi_dir))+'\n')
    target.write('chi^2 mag = '+str("{:.3f}".format(chi_mag))+'\n')
    target.close()
    
    return

def print_bfield(filename, out_dir, t_rope, btot_fit, br_fit, bt_fit, bn_fit, vel_fit, t_start, h):
    """Spit out the synthetic time series"""
    
    if (h==1):
        hs = 'p'
    elif (h==-1):
        hs = 'n'
        
    if isinstance(t_start, datetime.datetime):
        newt = []
        for tt in range(0,len(t_rope)):
            newt.append(t_start + datetime.timedelta(hours=t_rope[tt]))
        t_rope = np.asarray(newt)
    
    theta = np.array(np.rad2deg(np.arctan2(bn_fit, np.sqrt(br_fit**2+bt_fit**2))))
    phi = np.array(np.rad2deg(np.arctan2(bt_fit, br_fit)))
    phi[phi<0] = phi[phi<0]+360
    
    target = open(os.path.join(out_dir, filename), 'w')    
   
    target.write('########################################################\n')
    target.write('# EXPANDING LUNDQUIST BFIELD RESULTS #\n')
    target.write('# Time\t\t\tBtot [nT]\tBr[nT]\tBt [nT]\tBn [nT]\ttheta [°]\tphi [°]\tV [km/s]\n')
    
    for ii in range(0,len(t_rope)):
        target.write(str(t_rope[ii])+'\t'+str("{:.3f}".format(btot_fit[ii]))+'\t\t'+
            str("{:.3f}".format(br_fit[ii]))+'\t'+str("{:.3f}".format(bt_fit[ii]))+'\t'+str("{:.3f}".format(bn_fit[ii]))+'\t'+
            str("{:.3f}".format(theta[ii]))+'\t\t'+str("{:.3f}".format(phi[ii]))+'\t'+str("{:.3f}".format(vel_fit[ii]))+'\n')
    
    target.close()

    return

def create_fit_figure(result):
    """Create a publication-quality figure for the fitting results
    
    Args:
        result: Dictionary with fitting results
    
    Returns:
        Figure object
    """
    # Extract data from results
    t_rope = result['fit_data']['time']
    btot_fit = result['fit_data']['btot_fit']
    br_fit = result['fit_data']['br_fit']
    bt_fit = result['fit_data']['bt_fit']
    bn_fit = result['fit_data']['bn_fit']
    vel_fit = result['fit_data']['vel_fit']
    
    btot = result['observed_data']['btot']
    br = result['observed_data']['br']
    bt = result['observed_data']['bt']
    bn = result['observed_data']['bn']
    vel = result['observed_data']['vel']
    
    # Parameters
    params = result['optimized_parameters']
    derived = result['derived_parameters']
    
    # Create figure
    fig, axarr = plt.subplots(5, sharex=True, figsize=(10, 12))
    plt.subplots_adjust(hspace=.001) 

    helicity_sign = "+" if params['h'] == 1 else "-"
    title = f'Lundquist Flux Rope Fit (h={helicity_sign})'
    axarr[0].set_title(title, fontsize='x-large')
    
    max_time = max(t_rope)
    axarr[0].set_xlim(-1, max_time+1)

    # Plot |B|
    axarr[0].set_ylabel('|B| [nT]', fontsize='x-large')
    axarr[0].plot(t_rope, btot, color='black')
    axarr[0].plot(t_rope, btot_fit, color='magenta', lw=3)

    # Plot BR
    axarr[1].set_ylabel(r'B$_\mathrm{R}$ [nT]', fontsize='x-large')
    axarr[1].plot(t_rope, br, color='black')
    axarr[1].plot(t_rope, br_fit, color='magenta', lw=3)
    
    # Plot BT
    axarr[2].set_ylabel(r'B$_\mathrm{T}$ [nT]', fontsize='x-large')
    axarr[2].plot(t_rope, bt, color='black')
    axarr[2].plot(t_rope, bt_fit, color='magenta', lw=3)
    
    # Plot BN
    axarr[3].set_ylabel(r'B$_\mathrm{N}$ [nT]', fontsize='x-large')
    axarr[3].plot(t_rope, bn, color='black')
    axarr[3].plot(t_rope, bn_fit, color='magenta', lw=3)
    
    # Plot Velocity
    axarr[4].set_ylabel(r'V [km$\cdot$s$^{-1}$]', fontsize='x-large')
    axarr[4].plot(t_rope, vel, color='black')
    axarr[4].plot(t_rope, vel_fit, color='magenta', lw=3)

    axarr[4].set_xlabel(r'Time [h]', fontsize='x-large')
    
    # Add parameter info as text
    param_text = (
        f"Axis: θ = {params['theta0']:.1f}°, φ = {params['phi0']:.1f}°\n"
        f"p0 = {params['p0']:.3f}, B0 = {params['b0']:.2f} nT\n"
        f"R0 = {derived['r0']:.3f} AU, χ² = {derived['chi_dir']:.3f}"
    )
    axarr[0].text(0.02, 0.05, param_text, transform=axarr[0].transAxes, 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    fig.align_ylabels(axarr)

    for ax in axarr:
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.yaxis.set_ticks_position('both')
    
    fig.tight_layout()
    
    return fig

def perform_lundquist_fit(gui_data, region_datetime, parameters, output_dir):
    """Perform Lundquist fitting from GUI data
    
    Args:
        gui_data: Dictionary with mf and sw data from GUI
        region_datetime: Dictionary with region timestamps
        parameters: Dictionary with initial guess parameters
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with results and figures
    """
    try:
        # Step 1: Extract data from GUI data
        time, btot, br, bt, bn, vel, den, tem, first_time = load_from_gui_data(gui_data, region_datetime)
        
        # Define region bounds for fitting
        rope_bounds = [0, time[-1]]
        
        # Extract the data within the flux rope (already filtered, so just using all data)
        t_rope, br_rope, bt_rope, bn_rope, vel_rope = extract(rope_bounds, time, br, bt, bn, vel)
        
        # Set up the initial parameters as an lmfit Parameters object
        fparam = lmfit.Parameters()
        fparam.add('theta0', value=parameters['theta0'], min=-90, max=90)
        fparam.add('phi0', value=parameters['phi0'], min=0, max=360)
        fparam.add('p0', value=parameters['p0'], min=-1, max=1)
        fparam.add('h', value=parameters['h'], vary=False)
        fparam.add('b0', value=parameters['b0'], min=0)
        fparam.add('t0', value=parameters['t0'], min=0)
        
        # Set up list with data
        data = [br_rope, bt_rope, bn_rope, vel_rope]
        
        # Calculate initial chi squared values for comparison
        btot_fit_initial, br_fit_initial, bt_fit_initial, bn_fit_initial, vel_fit_initial = fit_initial_guess(fparam, t_rope, data)
        chi_dir_initial, chi_mag_initial = get_chi(br_rope, bt_rope, bn_rope, br_fit_initial, bt_fit_initial, bn_fit_initial, parameters['b0'])
        
        print('INITIAL CHIs', chi_dir_initial, chi_mag_initial)
        
        # Perform optimization
        solution = lmfit.minimize(fit_forever, params=fparam, method='leastsq', 
                                 args=(t_rope,), kws={'data': data, 'keyword': 'all'})
        
        # Extract final parameters from solution
        newpars = solution.params
        
        # Get the rope mean speed
        vel_avg = get_vel(vel_rope, t_rope)
        
        # Get the corresponding r0
        r0 = get_cloud_radius(t_rope, vel_avg, newpars['theta0'].value, newpars['phi0'].value, newpars['p0'].value)
        
        # Get the unit vectors of the cloud frame
        xc_hat, yc_hat, axis = rotate_coords(newpars['theta0'].value, newpars['phi0'].value)

        # Transform the RTN data into cloud frame
        br_cloud, bt_cloud, bn_cloud = transform_cloud_frame(xc_hat, yc_hat, axis, br_rope, bt_rope, bn_rope)

        # Get the measurement points through the cloud
        x_cross, samples = get_sampling_points(t_rope, newpars['p0'].value, r0, newpars['t0'].value)

        # Get the fitted flux rope Bfield
        b_axial, b_tangent = get_field_components(t_rope, newpars['t0'].value, newpars['b0'].value, newpars['h'].value, samples)

        # Transform back to RTN
        btot_fit, br_fit, bt_fit, bn_fit = cloud_to_cartesian(b_axial, b_tangent, xc_hat, yc_hat, axis, newpars['p0'].value, x_cross, samples)
        
        # Get modelled speed
        vel_fit = get_modelled_speed(t_rope, samples, x_cross, vel_avg, newpars['t0'].value, r0, newpars['p0'].value)
        
        # Calculate final chi squared values
        chi_dir, chi_mag = get_chi(br_rope, bt_rope, bn_rope, br_fit, bt_fit, bn_fit, newpars['b0'].value)
        
        print('FINAL CHIs', chi_dir, chi_mag)
        print(lmfit.fit_report(solution))
        
        # Calculate derived quantities - r0 was already calculated above
        flu_axis, flu_polo = get_rope_fluxes(newpars['b0'].value, r0)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename base
        filename_base = f"lundquist_fit_{region_datetime['mo_start'].strftime('%Y%m%d_%H%M')}"
        
        # Create result object to return
        result = {
            'optimized_parameters': {
                'theta0': newpars['theta0'].value,
                'phi0': newpars['phi0'].value,
                'p0': newpars['p0'].value,
                'h': newpars['h'].value,
                'b0': newpars['b0'].value,
                't0': newpars['t0'].value,
            },
            'derived_parameters': {
                'r0': r0,
                'flu_axis': flu_axis,
                'flu_polo': flu_polo,
                'chi_dir': chi_dir,
                'chi_mag': chi_mag
            },
            'fit_data': {
                'time': t_rope,
                'btot_fit': btot_fit,
                'br_fit': br_fit,
                'bt_fit': bt_fit,
                'bn_fit': bn_fit,
                'vel_fit': vel_fit
            },
            'observed_data': {
                'time': t_rope,
                # Use pre-sliced data directly:
                'btot': np.sqrt(br_rope**2 + bt_rope**2 + bn_rope**2),  # Calculate from components
                'br': br_rope,
                'bt': bt_rope,
                'bn': bn_rope,
                'vel': vel_rope,
            },
            'start_time': first_time
        }
        
        # Generate visualization
        fig = create_fit_figure(result)
        result['figure'] = fig
        
        # Save results to file
        print_params(f"{filename_base}.txt", output_dir, 
            newpars['theta0'].value, newpars['phi0'].value, newpars['p0'].value, 
            newpars['h'].value, newpars['b0'].value, newpars['t0'].value,
            r0, t_rope, flu_axis, flu_polo, chi_dir, chi_mag)
        
        # Save fit data
        print_bfield(f"{filename_base}_bfield.txt", output_dir,
                   t_rope, btot_fit, br_fit, bt_fit, bn_fit, vel_fit, first_time, newpars['h'].value)
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{filename_base}_fit.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Lundquist fitting: {str(e)}")
        logger.error(traceback.format_exc())
        raise
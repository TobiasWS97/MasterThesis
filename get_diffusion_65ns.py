import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.msd import EinsteinMSD
import pandas as pd
from scipy.stats import linregress
import glob
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import ruptures as rpt

def anomalous_diffusion(t, K_alpha, alpha):
        return K_alpha * t**alpha

def get_diffusion_labels(msd_values, times, r_value, change_points):
    # Only proceed if the R^2 value is over 0.90
    if r_value**2 > 0.80:
        labels = []
        # Start from the first change point
        for start, end in zip(change_points[:-1], change_points[1:]):
            slope = (msd_values[end] - msd_values[start]) / (times[end] - times[start])
            if slope < 1:
                labels.append('subdiffusive')
            elif slope > 1:
                labels.append('superdiffusive')
            else:
                labels.append('normal')
        return labels
    else:
        print("R^2 value is not over 0.90, skipping this simulation.")
        return None


def detect_change_points(data):
    model ="l2"
    algo = rpt.Binseg(model=model).fit(data)
    #change_points = algo.predict(pen=np.log(len(data)) * dim * sigma**2)
    change_points = algo.predict(n_bkps=2)
    return change_points

def calculate_diffusion_coefficient(topo_file, traj_file, selection, output_folder, folder_name):
    print(f"Processing {traj_file} with topology {topo_file}")
    start_time = time.time()
    
    try:
        u = mda.Universe(topo_file, traj_file)
        print("Loaded files.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None
    
    try:
        atom_selection = u.select_atoms(selection)
        print(f"Selected atoms: {len(atom_selection)}")
    except Exception as e:
        print(f"Error selecting atoms: {e}")
        return None, None

    if len(atom_selection) == 0:
        print("No atoms selected, skipping this file.")
        return None, None

    print("Starting MSD calculation.")
    # Compute the Mean Squared Displacement (MSD)
    try:
        msd = EinsteinMSD(atom_selection, select='all', msd_type='xyz', fft=True)
        msd.run()
        print("Calculated MSD.")
    except Exception as e:
        print(f"Error calculating MSD: {e}")
        return None, None

    print("Accessing MSD results.")
    # Get the MSD values and time
    try:
        msd_values = msd.results.timeseries
        times = np.arange(len(msd_values)) * u.trajectory.dt / 1000  # convert from ps to ns
        if len(msd_values) == 0 or len(times) == 0:
            print("MSD values or times are empty.")
            return None, None
        print(f"MSD values: {msd_values[:5]}...")  # Show first 5 values for brevity
        print(f"Times: {times[:5]}...")
    except Exception as e:
        print(f"Error accessing MSD results: {e}")
        return None, None
    
    print("Performing linear regression.")
    # Perform linear regression to obtain the slope
    try:
        s_time = 1.0  # 1 ns
        filtered_indices = (times > s_time)
        
        # Save the filtered times and msd_values to new variables
        times_filtered = times[filtered_indices]
        msd_values_filtered = msd_values[filtered_indices]

        slope, intercept, r_value, p_value, std_err = linregress(times_filtered, msd_values_filtered)
        print(f"Performed linear regression: slope = {slope}, intercept = {intercept}, r^2 = {r_value**2}")
    except Exception as e:
        print(f"Error in linear regression: {e}")
        return None, None

    print("Calculating diffusion coefficient.")
    # Diffusion coefficient D = slope / (2 * dimension_factor) * 1e-20 / 1e-12
    try:
        D = slope / (2 * msd.dim_fac) * 1e-20 / 1e-12  # converting to m²/s
        print(f"Calculated diffusion coefficient: D = {D}")
    except Exception as e:
        print(f"Error calculating diffusion coefficient: {e}")
        return None, None
    
    print("Fitting to anomalous diffusion equation.")
    # Perform linear regression to obtain the slope
    
    try:
        # t_values and msd_values should be your arrays of time and MSD values
        popt, pcov = curve_fit(anomalous_diffusion, times_filtered, msd_values_filtered)
        K_alpha_unconv, alpha = popt
        K_alpha = K_alpha_unconv * 1e-20 / 1e-12
    except Exception as e:
        print(f"Error in fit: {e}")
        return None, None
    
    print("Calculating R^2 and stderr for anomalous diffusion fit.")
    try:
        # Calculate the predicted MSD values for the actual time values
        msd_pred = anomalous_diffusion(times_filtered, K_alpha_unconv, alpha)

        # Calculate the R² value
        r2_anomalous = r2_score(msd_values_filtered, msd_pred)

        print(f'R² value for the anomalous diffusion fit: {r2_anomalous:.2f}')
        
        # Calculate stderr
        perr = np.sqrt(np.diag(pcov))
        K_alpha_err, alpha_err = perr

    except Exception as e:
        
        print(f"Error in calculating R^2 for anomalous diffusion fit: {e}")
        return None, None
    
    """print("Finding change points and labelling diffusion.")
    try:
        # Detect change points in the MSD values
        change_points = detect_change_points(msd_values_filtered)
        # Get diffusion labels
        diffusion_labels = get_diffusion_labels(msd_values_filtered, times_filtered, r_value, change_points) 
    except:
        print(f"Error in finding change points and diffusion areas: {e}")
        return None, None
        """
    
    print(f"Finished processing {traj_file} in {time.time() - start_time:.2f} seconds.")

    print("Plotting MSD and saving as PNG.")
    # Plot MSD and save as PNG
    try:
        plt.figure()
        # Add the linear regression line
        linear_fit = intercept + slope * times_filtered
        plt.plot(times_filtered, linear_fit, label='Linear fit')

        # Add the anomalous diffusion fit line
        anomalous_fit = K_alpha_unconv * times_filtered**alpha
        plt.plot(times_filtered, anomalous_fit, label='Anomalous fit')
        plt.plot(times_filtered, msd_values_filtered, label='MSD') 
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD ($Å^2$)')
        plt.title('Mean Squared Displacement')
        plt.legend()
        output_path = os.path.join(output_folder, "nvt_" + folder_name + '.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot as {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    print("Plotting MSD on log-log scale and saving as PNG.")
    # Plot MSD and save as PNG
    try:
        plt.figure()
        plt.plot(times_filtered, linear_fit, label='Linear fit')
        plt.plot(times_filtered, anomalous_fit, label='Anomalous fit')
        plt.plot(times_filtered, msd_values_filtered, label='MSD')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD ($Å^2$)')
        plt.title('Mean Squared Displacement')
        plt.legend()
        directory, filename = os.path.split(output_path)
        log_filename = "log-scale_" + filename
        log_output_path = os.path.join(directory, log_filename)
        plt.savefig(log_output_path)
        plt.close()
        print(f"Saved plot as {log_output_path}")
    except Exception as e:
        print(f"Error saving log-log plot: {e}")
    
    print("Plotting diffusion region of MSD.")
    try:
        plt.figure()
        plt.plot(times_filtered, msd_values_filtered, label='MSD')
        change_points = detect_change_points(msd_values_filtered)
        for cp in change_points[:-1]:
            plt.axvline(x=times_filtered[cp], color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('MSD')
        plt.title('MSD with Anomalous Regions')
        directory, filename = os.path.split(output_path)
        region_filename = "diffusion_region_" + filename
        region_output_path = os.path.join(directory, region_filename)
        plt.savefig(region_output_path)
        plt.close()
        print(f"Saved plot as {region_output_path}")
        
    except Exception as e:
        print(f"Error saving diffusion region plot: {e}")

    return D, r_value**2, std_err, p_value

def process_all_measurements(base_folder, tpr_folder, output_folder_plots, output_folder_CSV):
    results = []

    for value_folder in ['high_value', 'normal_value', 'low_value']:
        folder_path = os.path.join(base_folder, value_folder)

        for traj_file in glob.glob(os.path.join(folder_path, "*.xtc")):
            temp = os.path.splitext(os.path.basename(traj_file))[0]
            tpr_file = os.path.join(folder_path, temp + ".tpr") # if .tpr-file is in another folder use tpr_folder as first arg
            
            if not os.path.exists(tpr_file):
                print(f"TPR file {tpr_file} does not exist. Skipping this file.")
                continue

            try:
                D, r2, std_err, p_value = calculate_diffusion_coefficient(tpr_file, traj_file, "name Li", output_folder_plots, value_folder)
                print(D, r2)
                if D is not None and r2 is not None and std_err is not None and p_value is not None:
                    results.append({
                        'Original value': value_folder,
                        'Diffusion Coefficient': D,
                        'R^2': r2,
                        'stdandard error': std_err,
                        'p-value': p_value
                    })
                    print(results)
                else:
                    print(f"Skipping {traj_file} due to errors in calculation.")
            except Exception as e:
                print(f"Failed to process {traj_file}: {e}")

        # Save results to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(output_folder_CSV, 'diffusion_coefficients_65nsjob.csv'), index=False)
            print("Results saved to diffusion_coefficients_65nsjob.csv")
        else:
            print("No results to save.")

# Example usage
base_folder = "C:\\Users\\Bruker\\OneDrive - NTNU\\Y5\\Master_thesis\\Idun\\65ns_job"
tpr_folder = "C:\\Users\\Bruker\\OneDrive - NTNU\\Y5\\Master_thesis\\Idun\\65ns_job"
output_folder_plots = "C:\\Users\\Bruker\\OneDrive - NTNU\\Y5\\Master_thesis\\Idun\\65ns_job\\output_plots_diff_labels"
output_folder_CSV = "C:\\Users\\Bruker\\OneDrive - NTNU\\Y5\\Master_thesis\\Idun\\65ns_job\\output_CSV_diff_labels"

# Ensure output folder exists
if not os.path.exists(output_folder_plots):
    os.makedirs(output_folder_plots)

if not os.path.exists(output_folder_CSV):
    os.makedirs(output_folder_CSV)

process_all_measurements(base_folder, tpr_folder, output_folder_plots, output_folder_CSV)

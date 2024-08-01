import pandas as pd
import matplotlib.pyplot as plt
import os

# Set a consistent plotting style
plt.style.use('seaborn-v0_8-darkgrid')  
plt.rcParams['figure.figsize'] = [10, 6]  # Set a consistent figure size

# Load the CSV file with results from diffusion coefficient calculation into a pandas DataFrame
file_path = r'C:\Users\Bruker\OneDrive - NTNU\Y5\Master_thesis\Idun\MachineLearning\Data\diffusion_coefficients_cp_BinSeg_l1.csv'
df = pd.read_csv(file_path)

# Define the output directory
output_dir = r'C:\Users\Bruker\OneDrive - NTNU\Y5\Master_thesis\Idun\MachineLearning\Plots\diffusion_coefficients_BinSeg_l1_CORRECT_STYLE'
os.makedirs(output_dir, exist_ok=True)

# Filter data by concentrations
concentrations = [66, 75, 100]
colors = ['blue', 'green', 'red']
markers = ['o', 'x', 's']  # Circle, Cross, Square

# Log-scale plot, individual subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for ax, concentration, color, marker in zip(axes, concentrations, colors, markers):
    subset = df[df['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker)
    ax.set_yscale('log')
    ax.set_title(f'Concentration: {concentration} %')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Diffusion Coefficient (cm²/s)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_log-scale_INDIVIDUAL.png'))
plt.close()

# Filter out measurements with R^2 lower than 0.90
df_filtered = df[df['R^2'] >= 0.90]

# Set the figure size to 10x6 inches
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

# Define the y-axis limits based on the data
y_min = df_filtered['Diffusion Coefficient'].min()
y_max = df_filtered['Diffusion Coefficient'].max()

for ax, concentration, color, marker in zip(axes, concentrations, colors, markers):
    subset = df_filtered[df_filtered['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker)
    ax.set_yscale('log')
    ax.set_title(f'Concentration: {concentration} %')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Diffusion Coefficient (cm²/s)')
    ax.set_ylim([y_min, y_max])  # Set the same y-axis limits for all subplots

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_log-scale_with_R2_cutoff_0.9_INDIVIDUAL.png'))
plt.close()

# Non log-scale, individual subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for ax, concentration, color, marker in zip(axes, concentrations, colors, markers):
    subset = df[df['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker)
    ax.set_title(f'Concentration: {concentration} %')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Diffusion Coefficient (cm²/s)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_FULL_INDIVIDUAL.png'))
plt.close()

# Log-scale plot, combined
fig, ax = plt.subplots(figsize=(10, 6))

for concentration, color, marker in zip(concentrations, colors, markers):
    subset = df[df['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker, label=f'{concentration}%')
    
ax.set_yscale('log')
ax.set_title('Diffusion Coefficient vs Temperature (Log-Scale)')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Diffusion Coefficient (cm²/s)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_log-scale_COMBINED.png'))
plt.close()

# Log-scale plot with R^2 cutoff, combined
fig, ax = plt.subplots(figsize=(10, 6))

for concentration, color, marker in zip(concentrations, colors, markers):
    subset = df_filtered[df_filtered['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker, label=f'{concentration}%')
    
ax.set_yscale('log')
ax.set_title('Diffusion Coefficient vs Temperature')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Diffusion Coefficient (cm²/s)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_log-scale_with_R2_cutoff_0.9_COMBINED.png'))
plt.close()

# Non log-scale plot, combined
fig, ax = plt.subplots(figsize=(10, 6))

for concentration, color, marker in zip(concentrations, colors, markers):
    subset = df[df['Concentration'] == concentration]
    ax.scatter(subset['Temperature'], subset['Diffusion Coefficient'], color=color, marker=marker, label=f'{concentration}%')
    
ax.set_title('Diffusion Coefficient vs Temperature')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Diffusion Coefficient (cm²/s)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diff_coeff_VS_temp_COMBINED.png'))
plt.close()
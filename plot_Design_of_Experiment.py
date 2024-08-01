import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent plotting style
plt.style.use('seaborn-v0_8-darkgrid')  
plt.rcParams['figure.figsize'] = [10, 6]  # Set a consistent figure size


# Replace 'your_data_file.csv' with the path to your actual CSV file
file_path = r'C:\Users\Bruker\OneDrive - NTNU\Y5\Master_thesis\Idun\Design_of_Experiment\LHC_cat_1000.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first few rows to understand the data structure
print("Data Head:\n", data.head())

# Display the data types to ensure they are as expected
print("\nData Types:\n", data.dtypes)

# Assuming the CSV has columns named 'Temperature' and 'Class'
temperature_column = 'Temperature'
class_column = 'Class'

# Convert temperature to numeric if it's not already
data[temperature_column] = pd.to_numeric(data[temperature_column], errors='coerce')

# Convert class to string to map properly
data[class_column] = data[class_column].astype(str)

# Map the class values to concentration labels
class_mapping = {
    '1.0': '100%',
    '2.0': '75%',
    '3.0': '66%'
}

# Apply the mapping to the class column
data['Concentration'] = data[class_column].map(class_mapping)

# Check the mapping results
print("\nMapped Concentration Column:\n", data['Concentration'].head())

# Ensure concentration is a categorical type with the right order
data['Concentration'] = pd.Categorical(data['Concentration'], categories=['100%', '75%', '66%'], ordered=True)

# Display final data to be plotted
print("\nFinal Data:\n", data.head())

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Concentration', y=temperature_column, data=data, palette={'100%': 'red', '75%': 'green', '66%': 'blue'})
#plt.title('Temperature vs. Li Concentration')
plt.xlabel('Li Concentration')
plt.ylabel('Temperature (K)')

plt.grid(True)
plt.savefig("LHC_1000_swarmplot.png")
plt.show()

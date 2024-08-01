using LatinHypercubeSampling
using Plots
import Random
using CSV, DataFrames

Random.seed!(24)

# This is the number of samples 
numPoints = 1000


catWeight = 0.0
gens = 100

# Creating the dimensions with 3 categories
dims = [Continuous(),Categorical(3,catWeight)]

# Random samples for the previous dime
# Warning: The random numbers are Int64. I tried changing to Float64 and 32
#           and then using that in the next step, but I found and error
initialSample = randomLHC(numPoints,dims)

# Here is the modified LHS
X = LHCoptim!(initialSample,gens;dims=dims)[1]

# Collecting the data to plot
x_1 = X[X[:, 2] .== 1, 1]
x_2 = X[X[:, 2] .== 2, 1]
x_3 = X[X[:, 2] .== 3, 1]
index = 1:length(x_1)

scatter( index, x_1, markersize=4, xlabel="Index", ylabel="Range", title="Plot", label="")
scatter!(index, x_2, markersize=4, label="")
scatter!(index, x_3, markersize=4, label="")

# Changing the first column for the range of temperature
T_min = 258.0
T_max = 358.0

X_new = Float64.(X)
X_new[:,1] = (X_new[:,1] .- 1)./(numPoints-1)

X_new[:,1] = X_new[:,1].*(T_max - T_min) .+ T_min

# Collecting the new data to plot
x_1 = X_new[X_new[:, 2] .== 1, 1]
x_2 = X_new[X_new[:, 2] .== 2, 1]
x_3 = X_new[X_new[:, 2] .== 3, 1]
index = 1:length(x_1)

scatter( index, x_1, markersize=4, xlabel="Index", ylabel="Temperature", title="Categorical LHS", label="")
scatter!(index, x_2, markersize=4, label="")
scatter!(index, x_3, markersize=4, label="")

# Saving the data

Final_data = DataFrame(X_new, [:Temperature, :Class])
file = "LHC_cat_1000.csv"

CSV.write(file, Final_data)
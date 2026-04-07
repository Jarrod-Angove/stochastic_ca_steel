# A gpu-accelerated stochastic cellular automata model of microstructure development

This package provides a toy model for stochastic CA for microstructure evolution during arbitrary thermal processing. 

**Please note that this will only work on devices with AMD GPUs**. 

[![Benchmark simulation example](https://img.youtube.com/vi/2Z2rwcoDvuE/0.jpg)](https://www.youtube.com/watch?v=2Z2rwcoDvuE)

It leverages `AMDGPU.jl` via custom kernels to do CA simulations in as little as 7 seconds for 1000x1000 domains over >30 minutes of simulated process. 

Please see the attached report for the theory behind the model. 

## How to run the model

After pulling the repository, navigate to its folder and activate the Julia module in a REPL. This might take a while as it might need to precompile `GLMakie.jl`. 

```julia
using Pkg
Pkg.activate(".")
using kmc_gpu
```

The simulation runs through a single function call to `track_state_over_time_gpu`. It has the following fields: 

```julia
track_state_over_time_gpu(
S_init,                 # An NxM array of touples, where each touple contains the phase (0 or 1) and theta (0, 10, 20, ..., 90) determines initial conditions
t_array,                # A vector with points in time where the model will be forced to evaluate and save results
T_func,                 # A function that determines the temperature (in Kelvin) for any point in time
params;                 # A `SimParams` struct that contains parameters like mobility and acitvation energy (see below)
Δt_floor=1f-6           # Smallest allowable time step (prevents the simulation time from exploding if kinetics are too fast)
plot_every=10,          # An option that tells the simulation how frequently it should save a frame for a video (i.e. every 10 macro-steps by default)
record_video=true,      # If disabled, no plotting will be done but microstructure stats will still be saved (massive performance boost)
show_debug_stats=false, # Will print some info regarding max free energy etc if enables
benchmark=false,        # If true, this disables record_video, and prevents the creation of a simulation directory to prevent clutter during benchmarking
)
```

This function will take an initial state, a set of parameters, and a thermal history. It will create a simulation results folder in your working directory, where it will 
save the time series for microstructure statistics and a corresponding video of the microstructure development. 

There are some utilities to help initialize a simulation. For example, `initialize_real_microstructure(...)` can generate the `S_init` field via a vornoi algorithm: 

```julia
S_init = initialize_real_microstructure(
N,            # Number of columns in the microstructure
M,            # Number of rows
num_grains;   # How many grains in the microstructure
phase=1)      # What is the initial phase (i.e. austenite = 0 for high temp initialization or ferrite = 1 for low temp initialization)
```

There is also a function `thermal_cycle_func` that provides a cosine thermal history for easy testing (but you can use any julia function 
that returns a scalar). 

```julia
thermal_cycle_func(t;
T_max=1053,    # Maximum temperature the cycle will reach in kelvin. This will also be the start temp. 
T_min=773,     # Minimum temperature of the cycle in kelvin. 
period=900)    # Period of the cycle in seconds.
```

To run a simulation, you will also need to select the parameters. This is done using the `SimParams` struct: 

```julia
struct SimParams
    inter_factor::Float32 # interphase surface energy orientation dependence factor (unitless)
    F_factor::Float32     # ferrite-ferrite interface orientation dependence factor (ev/um^2)
    A_factor::Float32     # austenite-austenite interface orientation dependece factor (ev/um^2)
    gamma_inter::Float32  # baseline interphase surface energy (ev/um^2)
    Q_mobility::Float32   # Activation energy for mobility factor (J/mol)
    M_0::Float32          # Pre-exponential factor for mobility (m^4/(J s))
    K_nuc::Float32        # Nucleation rate constant
    l::Float32            # Length of a single pixel in microns
    P_max::Float32        # limit on transition prob; passing this will cause smaller timesteps
    θ_threshold::Float32  # threshold for what is considered a high angle grain boundary
    M_GB_multiplier::Float32  # Mobility multiplier for fast diffusion at grain boundaries
end
```

The values used in the attached report are: 

```julia
default_params = SimParams(
           0.9f0,
           4.5f6,
           4.1f6,
           7.0f6,
           1.7f5,
           1.8f-4,
           6.0f-6,
           0.5f0,
           0.5f0, 20f0, 6f0
       )
```

## Typical workflow

After activating and `using` the `kmc_gpu` package, you can start by creating a time vector, a thermal history function, and a set of simulation parameters: 

```julia
t = collect(0:1:2400.0)
T_func(t) = thermal_cycle_func(t)
default_params = SimParams(
           0.9f0,
           4.5f6,
           4.1f6,
           7.0f6,
           1.7f5,
           1.8f-4,
           6.0f-6,
           0.5f0,
           0.5f0, 20f0, 6f0
       )
```

You can then initialize a microstrucutre: 

```julia
S_init = initialize_real_microstructure(4000, 4000, 3000; phase=0)
```

If you would like to see this initial structure before starting the simulation, you can use the `tuple_to_rgb` function in conjunction with the Makie library; 

```julia
using GLMakie
heatmap(tuple_to_rgb.(S_init))
```

If you are happy with the initial microstructure, all that is left to do is run the simulation function; 

```julia
S_out, results = track_state_over_time(S_init, t, T_func, default_params; plot_every=2)
```

This will create a results directory at your local working directory. The `S_out` will be the final frame of the simulation. `results` is an array of vectors containing the time series statistics. 

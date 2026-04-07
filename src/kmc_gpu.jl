module kmc_gpu

using AMDGPU
using ProgressMeter
using GLMakie
using Colors
using Printf
using Dates
using DelimitedFiles

const kB = 8.617333262f-5 # Blotzmann constant in eV/K
const h̄ = 6.582119569f-16  # Planck constant in eV s
const R_gas = 8.314f0         # Universal gas constant (J/(mol*K))

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
    θ_threshold::Float32
    M_GB_multiplier::Float32
end

default_params = SimParams(
    0.1f0,
    5.1f6,
    6.1f6,
    5.23f6,
    1.7f5,
    2.0f-4,
    5.0f-7,
    1.0f0,
    0.15f0,
    20f0,
    2f0
)

function gibbs_ferrite(T::Float32)
    T_C = T - 273.0f0
    return -1.52f4 * T_C^2 - 3.33f7 * T_C - 4.26f9  # Gives eV/um^3
end

function gibbs_austenite(T::Float32)
    T_C = T - 273.0f0
    return -8.31f3 * T_C^2 - 4.83f7 * T_C + 3.12f9 # Gives eV/um^3
end

function abs_ΔG(T::Float32)
    return abs(gibbs_ferrite(T) - gibbs_austenite(T))
end

# Precalculated sin values for 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 degrees
const SIN_LOOKUP = (0.0f0, 0.173648f0, 0.34202f0, 0.5f0, 0.642787f0, 0.766044f0,
    0.866025f0, 0.939692f0, 0.984807f0, 1.0f0)

function interface_energy(dθ::Int32, p1::Int32, p2::Int32, params::SimParams)
    
    # Get the index (e.g., dθ = 20 -> 20/10 + 1 = index 3)
    idx = (dθ ÷ Int32(10)) + Int32(1)
    
    sin_val = SIN_LOOKUP[idx]
    
    ifelse(p1 != p2,
        params.gamma_inter * (1.0f0 + params.inter_factor * sin_val),
        ifelse(p1 == 1, 
            params.F_factor * sin_val, 
            params.A_factor * sin_val
        )
    ) 
end

# Total energy for a single voxel (thermodynamic free energy)
function total_vox_energy(T::Float32, θ::Int32, p::Int32,
                          θ_n::NTuple{4, Int32}, p_n::NTuple{4, Int32}, params::SimParams)
    l = params.l
    l2 = l * l
    l3 = l2 * l
    
    # Bulk energy
    G_v = ifelse(p == 0, gibbs_austenite(T), gibbs_ferrite(T)) * l3 

    sum_interface = 0.0f0
    
    sum_interface += interface_energy(abs(θ_n[1] - θ), p, p_n[1], params) * l2
    sum_interface += interface_energy(abs(θ_n[2] - θ), p, p_n[2], params) * l2
    sum_interface += interface_energy(abs(θ_n[3] - θ), p, p_n[3], params) * l2
    sum_interface += interface_energy(abs(θ_n[4] - θ), p, p_n[4], params) * l2

    return G_v + sum_interface
end

# Convert to a fixed-size Tuple of Tuples for the GPU
const POSSIBLE_STATES = Tuple((Int32(θ), Int32(p)) for θ in 0:10:90 for p in 0:1)

#const Q_mobility = 1.7e5    # Activation energy for boundary mobility (J/mol)
#const M_0 = 2.0f-4          # Pre-exponential mobility factor (m^4 / (J*s))
function get_M(T::Float32, params::SimParams)

    #l = params.l

    # 1. Get driving force in eV / um^3
    #dG_vol_eV = abs(gibbs_ferrite(T) - gibbs_austenite(T)) 
    
    # 2. Convert to SI units (J / m^3) for the mobility equation
    # 1 eV/um^3 = 0.160218 J/m^3
    #dG_vol_SI = dG_vol_eV * 0.160218f0 
    
    # 3. Calculate physical mobility (m^4 / (J*s))
    mobility = params.M_0 * exp(-params.Q_mobility / (R_gas * T))
    
    # 4. Calculate target velocity in SI units (m/s)
    #v_target_SI = mobility * dG_vol_SI
    
    # 5. Convert velocity to microns/sec (um/s) so it matches your grid
    #v_target_um = v_target_SI * 1.0f6
    
    # 6. Return attempt frequency (v / l) where l is in microns
    return mobility
end

function kmc_kernel!(S_new, S_old, R_matrix, Δt::Float32, T::Float32,
                                pass_parity::Int32, params::SimParams)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 1) * workgroupDim().y
    
    N, M = size(S_old)
    
    if i <= N && j <= M
        # --- RED-BLACK CHECKERBOARD LOGIC ---
        if (i + j) % 2 != pass_parity
            S_new[i, j] = S_old[i, j]
            return nothing
        end
        
        θ, p = S_old[i, j]
        
        il = ifelse(i == 1, N, i - 1)
        ir = ifelse(i == N, 1, i + 1)
        jt = ifelse(j == 1, M, j - 1)
        jb = ifelse(j == M, 1, j + 1)
        
        θl, pl = S_old[il, j] 
        θt, pt = S_old[i, jt]
        θr, pr = S_old[ir, j]
        θb, pb = S_old[i, jb]
        
        θ_n = (θl, θt, θr, θb)
        p_n = (pl, pt, pr, pb)
        
        current_energy = total_vox_energy(T, θ, p, θ_n, p_n, params)
        l = params.l
        
        candidates = ((θl, pl), (θt, pt), (θr, pr), (θb, pb))
        
        r = R_matrix[i, j] 
        
        cum_P = 0.0f0
        chosen_state = (θ, p) 
        
        for idx in 1:4
            cand_θ, cand_p = candidates[idx]
            
            is_current = (cand_θ == θ) && (cand_p == p)
            
            # Let probabilities stack physically.
            if !is_current 
                E_cand = total_vox_energy(T, cand_θ, cand_p, θ_n, p_n, params)
                
                # ΔE is the change in energy. Negative means it is thermodynamically favorable.
                ΔE = E_cand - current_energy 
                
                # In a macroscopic CA, boundaries only move if there is a thermodynamic driving force.
                # (Thermal fluctuations against the gradient are negligible at the micron scale).
                if ΔE < 0.0f0
                    # 1. Deterministic Velocity (v = M * Driving Force)
                    driving_force = -ΔE
                    # 1. Determine if the voxel (θ, p) being consumed sits on a grain boundary 
                    # of its OWN current phase. 
                    is_on_GB = false
                    θ_HA = params.θ_threshold # only high angle GBs are considered GBs

                    if (pl == p && abs(θ - θl) > Int32(θ_HA)) ||
                       (pt == p && abs(θ - θt) > Int32(θ_HA)) ||
                       (pr == p && abs(θ - θr) > Int32(θ_HA)) ||
                       (pb == p && abs(θ - θb) > Int32(θ_HA))
                        is_on_GB = true
                    end

                    # 2. Apply the global kinetic multiplier to the mobility
                    mobility = get_M(T, params)
                    if is_on_GB
                        mobility *= params.M_GB_multiplier 
                    end

                    # 3. Calculate the new, anisotropically-driven velocity
                    v = mobility * driving_force * 0.160218f6
                    # 2. Macroscopic Capture Probability
                    P_capture = (v * Δt) / l
                    
                    # Add this specific boundary's pull to the cumulative probability
                    cum_P += P_capture
                    
                    if r <= cum_P
                        chosen_state = candidates[idx]
                        break
                    end
                end
            end
        end
        
        S_new[i, j] = chosen_state
    end
    return nothing
end

function nucleate_kernel!(S, R_probs, R_seeds, Δt::Float32, T::Float32,
                          inv_kBT::Float32, params::SimParams)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 1) * workgroupDim().y

    N, M = size(S)
    if i <= N && j <= M
        θ, p = S[i, j]

        # Get the current neighbors
        il = ifelse(i == 1, N, i - 1)
        ir = ifelse(i == N, 1, i + 1)
        jt = ifelse(j == 1, M, j - 1)
        jb = ifelse(j == M, 1, j + 1)

        θl, pl = S[il, j] 
        θt, pt = S[i, jt]
        θr, pr = S[ir, j]
        θb, pb = S[i, jb]

        θ_n = (θl, θt, θr, θb)
        p_n = (pl, pt, pr, pb)

        # Propose a new state: flip the phase, and pick a random orientation
        new_p = ifelse(p == 0, Int32(1), Int32(0))
        # Ensure the random orientation maps to 0, 10, ..., 90 degree states
        new_θ = Int32((R_seeds[i, j] % 10) * 10) 

        # Calculate the strict thermodynamic energy change
        E_old = total_vox_energy(T, θ, p, θ_n, p_n, params)
        E_new = total_vox_energy(T, new_θ, new_p, θ_n, p_n, params)
        ΔE = E_new - E_old

        # Calculate the probability of this nucleation event occurring
        #base_rate = nu * exp(-Q_nuc * inv_kBT)
        
        # Metropolis probability: heavily penalizes nucleation in the bulk, 
        # but allows it on boundaries where ΔE is smaller or negative
        metropolis_prob = ifelse(ΔE <= 0.0f0, 1.0f0, exp(-ΔE * inv_kBT))
        
        # K_NUC is now purely an attempt frequency multiplier for spontaneous structural fluctuations
        freq = get_M(T, params) * abs(ΔE)/params.l  * 0.160218f6
        rate = params.K_nuc * freq * metropolis_prob
        P_trans = 1.0f0 - exp(-rate * Δt)

        if R_probs[i, j] < P_trans
            S[i, j] = (new_θ, new_p)
        end
    end
    return nothing
end

function find_max_velocity_kernel!(V_gpu, S_old, T::Float32, params::SimParams)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 1) * workgroupDim().y

    N, M = size(S_old)

    if i <= N && j <= M
        θ, p = S_old[i, j]
        
        il = ifelse(i == 1, N, i - 1)
        ir = ifelse(i == N, 1, i + 1)
        jt = ifelse(j == 1, M, j - 1)
        jb = ifelse(j == M, 1, j + 1)
        
        θl, pl = S_old[il, j]
        θt, pt = S_old[i, jt]
        θr, pr = S_old[ir, j]
        θb, pb = S_old[i, jb]
        
        θ_n = (θl, θt, θr, θb)
        p_n = (pl, pt, pr, pb)
        
        current_energy = total_vox_energy(T, θ, p, θ_n, p_n, params)
        candidates = ((θl, pl), (θt, pt), (θr, pr), (θb, pb))
        
        local_max_v = 0.0f0
        
# --- GENERALIZED GB DETECTION ---
        is_on_GB = false
        if (pl == p && abs(θ - θl) > Int32(0)) ||
           (pt == p && abs(θ - θt) > Int32(0)) ||
           (pr == p && abs(θ - θr) > Int32(0)) ||
           (pb == p && abs(θ - θb) > Int32(0))
            is_on_GB = true
        end
        
        mobility = get_M(T, params)
        if is_on_GB
            mobility *= params.M_GB_multiplier
        end
        # --------------------------------
        
        for idx in 1:4
            cand_θ, cand_p = candidates[idx]
            
            if (cand_θ != θ) || (cand_p != p)
                E_cand = total_vox_energy(T, cand_θ, cand_p, θ_n, p_n, params)
                ΔE = E_cand - current_energy
                
                if ΔE < 0.0f0
                    # Use the multiplied mobility to find the true max velocity
                    v = mobility * abs(ΔE) * 0.160218f6
                    local_max_v = max(local_max_v, v)
                end
            end
        end
        
# --- THE HIGH-PERFORMANCE FLOAT ATOMIC TRICK ---
        # Only attempt a memory write if the pixel is actively moving.
        # This drastically reduces atomic contention down to just the active phase boundaries.
        if local_max_v > 0.0f0
            # Reinterpret the Float32 bits as UInt32 to use hardware integer atomics
            val_uint = reinterpret(UInt32, local_max_v)
            
            # Use the modern AMDGPU macro syntax instead of the deprecated pointer function
            AMDGPU.@atomic max(V_gpu[1], val_uint)
        end
    end
    return nothing
end

function track_state_over_time_gpu(S_init, t_array, T_func, params::SimParams ;  
                                   Δt_floor = 1f-6,   #hard limit on smallest time step (s) 
                                   plot_every = 10,
                                   record_video = true, 
                                   framerate = 24,
                                   show_debug_stats=false, 
                                   smart_steps=true, 
                                   benchmark=false)

    l = params.l
    P_MAX = params.P_max

    if benchmark
        record_video = false
    end

    if ! benchmark
        # --- 1. Create Output Directory ---
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        out_dir = "sim_results_$timestamp"
        video_filename = "simulation_$timestamp.webm"
        mkpath(out_dir)
        println("Created output directory: $out_dir")

        # --- 2. Save Simulation Parameters to CSV ---
        params_file = joinpath(out_dir, "parameters.csv")
        open(params_file, "w") do io
            write(io, "Parameter,Value\n")
            for f in fieldnames(SimParams)
                val = getfield(params, f)
                write(io, "$f,$val\n")
            end
        end
    end

    N, M = size(S_init)
    n_steps = length(t_array) - 1
    
    S_init_32 = [Tuple{Int32, Int32}(s) for s in S_init]
    S_d1 = ROCArray(S_init_32)
    S_d2 = ROCArray(similar(S_init_32))
    
    vstream = nothing
    if record_video

        S_host = Array(S_d1) 
        inst_T = Observable(T_func(t_array[1]) - 273.0f0)
        inst_time = Observable(Float32(t_array[1])) 
        my_safe_t = Observable(Float32(t_array[2] - t_array[1]))
        # --- Visualization Setup ---
        f = Figure(size=(2560, 1440); fontsize=48)
        
        # Create observables for both Temperature and Time
        inst_T = Observable(T_func(t_array[1]) - 273.0f0)
        inst_time = Observable(Float32(t_array[1])) 
        my_safe_t = Observable(Float32(t_array[2] - t_array[1]))
        
        img = Observable(tuple_to_rgb.(S_host))
    
    # Lift on BOTH observables to update the title dynamically
        dynamic_title = lift(inst_T, inst_time, my_safe_t) do myT, myt, myst
        # %8.4f means: take up 8 total characters, with exactly 4 decimal places
        @sprintf("Time: %8.4f s  |  Temp: %7.2f °C  |  Δt_safe: %7.4f", myt, myT, myst)
        end
    
        ax = Axis(f[1,1], title=dynamic_title, aspect=DataAspect())
            image!(ax, img, interpolate=false) 
        # Initialize the Makie video stream attached to your figure
        vstream = VideoStream(f, framerate=framerate, compression=1)
    end

    # --- GPU Setup ---
    threads = (16, 16)
    blocks = (ceil(Int, N / threads[1]), ceil(Int, M / threads[2]))
    l2 = Float32(l * l)

    tracked_safe_times = Float32[]
    tracked_dt = Float32[]
    tracked_ferrite_frac = Float32[]
    tracked_phase_area = Float32[]
    tracked_diameter = Float32[]

    d_f_count = AMDGPU.zeros(Int32, N, M)
    d_pb_count = AMDGPU.zeros(Int32, N, M)
    d_gb_count = AMDGPU.zeros(Int32, N, M)

    R_nuc = AMDGPU.rand(Float32, N, M)
    R_seeds = AMDGPU.rand(UInt32, N, M)
    R_d_red = AMDGPU.rand(Float32, N, M)
    R_d_black = AMDGPU.rand(Float32, N, M)


    total_pixels = Float32(N * M)


    total_iters = 0
    
    # A single element to hold the max velocity bits
    max_v_gpu = AMDGPU.zeros(UInt32, 1)

    # --- Main Loop ---
    current_time = 0.0f0
    @showprogress for i in 1:n_steps
        Δt_macro = Float32(t_array[i + 1] - t_array[i])
        Δt_safe = Δt_macro  #initialize a value   
        Δt_actual = Δt_safe #initialize a value  
        current_T = Float32(T_func(current_time))
        inv_kBT = 1.0f0 / (kB * current_T)

        # Smart steps calculates the maximum velocity at every macro step 
        # to adaptively select Δt for the subsequent microsteps. 
        if smart_steps
            # 1. Reset the single global tracker
            AMDGPU.fill!(max_v_gpu, UInt32(0))

            # 2. Dispatch the reduction kernel
            @roc groupsize=threads gridsize=blocks find_max_velocity_kernel!(max_v_gpu,
                                                                             S_d1, current_T,
                                                                             params)
            AMDGPU.synchronize()

            # 3. Pull the UInt32 back to the CPU and translate it back to Float32
            max_v_uint = Array(max_v_gpu)[1]
            actual_max_v = reinterpret(Float32, max_v_uint)

            # 4. Handle the edge case where nothing is moving (v = 0)
            if actual_max_v == 0.0f0
                # If nothing wants to move, take the largest allowed macro step
                Δt_safe = Δt_macro 
            else
                # Calculate dynamic CFL limit based strictly on the fastest pixel RIGHT NOW
                Δt_safe = max(min(Δt_macro, (P_MAX * l) / actual_max_v), Δt_floor)
            end
        
            num_micro_steps = ceil(Int, Δt_macro / Δt_safe)
            Δt_actual = Δt_macro / Float32(num_micro_steps)
        else
            # If smart steps is off, calculate the maximum possible velocity 
            # accross ANY transformation for every time step
            # Largest possible boundary energy
            gamma_max = interface_energy(Int32(90f0), Int32(0), Int32(1), params)
            max_boundary_release = 4.0f0 * gamma_max * l^2
            G_ferrite = gibbs_ferrite(current_T)
            G_austenite = gibbs_austenite(current_T)
            max_chemical_release = abs(G_ferrite - G_austenite) * (l^3)

            # Largest possible chemical driving force
            max_driving_force = max_boundary_release + max_chemical_release

            # Max velocity
            max_v = get_M(current_T, params) * max_driving_force * 0.160218f6

            # Courant–Friedrichs–Lewy (CFL) limit for the Cellular Automaton:
            # P_capture = (v * Δt) / l. We ensure 4 * P_capture never exceeds P_MAX.
            Δt_safe = max(min(Δt_macro, (P_MAX * l) / (4.0f0 * max_v)), Δt_floor)
            num_micro_steps = ceil(Int, Δt_macro / Δt_safe)

            Δt_actual = Δt_macro / Float32(num_micro_steps)
            @show Δt_actual  
        end

        if Δt_safe < Δt_actual  
            @warn "Step size may be too large; \n Δt_safe=$Δt_safe    Δt_actual=$Δt_actual"
        end

        if show_debug_stats
            @show max_chemical_release
            @show max_boundary_release
            @show max_driving_force
            @show max_rate
            @show Δt_safe
            @show 1 - exp(-max_rate * Δt_actual)
        end

        # --- Calculate Stats on the GPU EVERY step ---
        @roc groupsize=threads gridsize=blocks compute_local_stats_kernel!(
            S_d1, d_f_count, d_pb_count, d_gb_count
        )
        AMDGPU.synchronize()
        
        # The GPU performs the reduction
        total_ferrite = sum(d_f_count)
        total_pb = sum(d_pb_count)
        total_gb = sum(d_gb_count)
        
        # Complete the final math on the CPU
        ferrite_fraction = Float32(total_ferrite) / total_pixels
        total_phase_area = Float32(total_pb) * l
        
        total_boundaries = Float32(total_pb + total_gb)
        if total_boundaries > 0.0f0
            P_L = total_boundaries / (2.0f0 * total_pixels * l)
            avg_diameter = 1.0f0 / P_L
        else
            avg_diameter = Float32(N * l)
        end
        
        push!(tracked_safe_times, Δt_safe)
        push!(tracked_dt, Δt_actual)
        push!(tracked_ferrite_frac, ferrite_fraction)
        push!(tracked_phase_area, total_phase_area)
        push!(tracked_diameter, avg_diameter)

        for _ in 1:num_micro_steps
        # --- 1. NUCLEATION PHASE ---
            AMDGPU.rand!(R_nuc)
            AMDGPU.rand!(R_seeds)
            T = Float32(T_func(current_time))

            @roc groupsize=threads gridsize=blocks nucleate_kernel!(
                S_d1, R_nuc, R_seeds, Δt_actual, T, inv_kBT, params
            )

            # --- 2. GROWTH PHASE (RED-BLACK CHECKERBOARD) ---
            # RED PASS (parity = 0)
            # Reads from S_d1, calculates updates for Red pixels, writes to S_d2
            # Black pixels are copied over identically
            AMDGPU.rand!(R_d_red)
            @roc groupsize=threads gridsize=blocks kmc_kernel!(
                S_d2, S_d1, R_d_red, Δt_actual, T, Int32(0), params
            )
            
            # BLACK PASS (parity = 1)
            # Reads from S_d2 (which now contains updated Reds), calculates updates
            # for Black pixels, writes back to S_d1
            AMDGPU.rand!(R_d_black)
            @roc groupsize=threads gridsize=blocks kmc_kernel!(
                S_d1, S_d2, R_d_black, Δt_actual, T, Int32(1), params
            )
            total_iters += 1 
            current_time += Δt_actual 
        end

        if record_video
            if i % plot_every == 0 || i == n_steps
                copyto!(S_host, S_d1) 
                
                # Update visual observables
                inst_T[] = current_T - 273.0f0
                inst_time[] = Float32(t_array[i+1]) 
                img[] = tuple_to_rgb.(S_host)
                my_safe_t[] = Δt_safe

                # Prevents skipped frames
                yield() 
                recordframe!(vstream)
            end
        end
    end

    @show total_iters
    if record_video
        full_vid_path = joinpath(out_dir, video_filename)
        println("Saving video to $video_filename...")
        save(full_vid_path, vstream)
        println("Video saved successfully!")
    end

    results = (
        times = tracked_safe_times,
        dt_sizes = tracked_dt,
        ferrite_fraction = tracked_ferrite_frac,
        phase_area = tracked_phase_area,
        avg_diameter = tracked_diameter
    )

    if ! benchmark
        # --- 3. Save Time-Series Results to CSV ---
        results_file = joinpath(out_dir, "timeseries_results.csv")
        open(results_file, "w") do io
            # Write the header row
            write(io, "Time_s,Temperature_C,Ferrite_Fraction, Phase_Area_um,Avg_Diameter_um,dt_safe,dt_actual\n")

            # Write the data rows
            for k in 1:n_steps
                t_val = t_array[k + 1]
                T_val = T_func(t_val) - 273.0f0 # Converted to Celsius for readability
                frac = tracked_ferrite_frac[k]
                p_area = tracked_phase_area[k]
                diam = tracked_diameter[k]
                dt_s = tracked_safe_times[k]
                dt_a = tracked_dt[k]
                
                write(io, "$t_val,$T_val,$frac,$p_area,$diam,$dt_s,$dt_a\n")
            end
        end
        println("Time-series results saved to $results_file")
        
        plot_simulation_results(results_file)
        plot_temperature_dependence(results_file)
    end
    return Array(S_d1), results
end

function tuple_to_rgb(state::Tuple{Int32, Int32})
    θ, p = state
    
    norm_θ = Float32(θ) / 90.0f0
    
    if p == 0 
        return HSV(norm_θ * 60.0f0, 0.8f0, 0.9f0) 
    else
        return HSV(200.0f0 + norm_θ * 60.0f0, 0.8f0, 0.9f0) 
    end
end

# instantiate a frame with random orientation for every pixel
function instantiate_random_microstructure(N, M)
    [(rand(collect(0:10:90)), rand([0,0])) for i in 1:N, j in 1:M]
end

function tuples_to_tensor(A)
    permutedims(stack(A), (2, 3, 1))
end

# Constant cooling/heating thermal history
function generate_T_t(cr, T0, Δt, Nt)
    t = 0:Δt:(Δt*Nt)
    @show Δt*Nt
    T = T0 .- cr .* t
    @show T[end]
    return [t, T]
end


function generate_cc(T0, Tf, cr, Nt)
    Δt = (T0 - Tf)/cr / Nt  
    t_f = Δt * Nt 
    t = 0:Δt:t_f
    T = T0 .- cr .* t
    return collect(t), collect(T)
end

function cc_func(t; T0=(780f0+273f0), Tf=(500f0 + 273f0), cr = 1f0)
    return ifelse(T0 - cr * t > Tf, T0 - cr * t, Tf)
end


function thermal_cycle(T_max, T_min, Δt, period, N_cycles)
    t = collect(0:Δt:(N_cycles*period))

    N_steps = length(t)
    t_final = t[end]
    @show N_steps
    @show t_final
    max_rate = (T_max - T_min)/2 * 2π/period 
    @show max_rate
    T = @. ((T_max - T_min) * cos(t/period * 2π) + (T_max + T_min))/2
    return t, T
end

function thermal_cycle_func(t; T_max = (780+273), T_min = 500 + 273, period = 900)::Float32
    return ((T_max - T_min) * cos(t/period * 2π) + (T_max + T_min))/2
end


# Generates a vornoi diagram type of microstructure with num_grains seeds
function initialize_real_microstructure(N::Int, M::Int, num_grains::Int; phase=1)
    # Initialize an empty host array with your precise Tuple types
    S_init = Array{Tuple{Int32, Int32}}(undef, N, M)
    
    # 1. Generate random coordinates for the grain "seeds"
    seed_x = rand(1:N, num_grains)
    seed_y = rand(1:M, num_grains)
    
    # 2. Assign a random orientation (0, 10, ..., 90) and Phase 1 (Ferrite) to each seed
    # You can change rand(0:10:90) if you ever want continuous orientations
    seed_states = [(Int32(rand(0:10:90)), Int32(phase)) for _ in 1:num_grains]
    
    # 3. Populate the grid using the closest seed 
    Threads.@threads for i in 1:N
        for j in 1:M
            min_dist_sq = Inf
            closest_seed = 1
            
            for k in 1:num_grains
                # Calculate distance in X and Y
                dx = abs(i - seed_x[k])
                dy = abs(j - seed_y[k])
                
                # Apply Periodic Boundary Conditions to the distance calculation
                # This ensures a grain on the right edge smoothly wraps to the left edge
                dx = min(dx, N - dx)
                dy = min(dy, M - dy)
                
                # Squared Euclidean distance (faster than taking the square root)
                dist_sq = dx^2 + dy^2
                
                if dist_sq < min_dist_sq
                    min_dist_sq = dist_sq
                    closest_seed = k
                end
            end
            
            # Assign the pixel the state of its closest seed
            S_init[i, j] = seed_states[closest_seed]
        end
    end
    
    return S_init
end

function compute_local_stats_kernel!(S, f_count, pb_count, gb_count)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 1) * workgroupDim().y
    
    N, M = size(S)
    if i <= N && j <= M
        θ, p = S[i, j]
        
        # 1. Map Ferrite count (1 if Ferrite, 0 otherwise)
        f_count[i, j] = ifelse(p == 1, Int32(1), Int32(0))
        
        # 2. Map Boundary counts
        p_edges = Int32(0)
        g_edges = Int32(0)
        
        ir = ifelse(i == N, 1, i + 1)
        jb = ifelse(j == M, 1, j + 1)
        
        θr, pr = S[ir, j]
        θb, pb = S[i, jb]
        
        # Evaluate Right neighbor
        if p != -1 && pr != -1
            if p != pr
                p_edges += Int32(1)
            elseif θ != θr
                g_edges += Int32(1)
            end
        end
        
        # Evaluate Bottom neighbor
        if p != -1 && pb != -1
            if p != pb
                p_edges += Int32(1)
            elseif θ != θb
                g_edges += Int32(1)
            end
        end
        
        pb_count[i, j] = p_edges
        gb_count[i, j] = g_edges
    end
    return nothing
end

function plot_simulation_results(csv_path::String)
    # Read the data, separating the header row from the actual numbers
    data, header = readdlm(csv_path, ',', header=true)
    
    # Extract the columns based on the 
    time_s = Float32.(data[:, 1])
    temp_C = Float32.(data[:, 2])
    ferrite_frac = Float32.(data[:, 3])
    avg_diameter = Float32.(data[:, 5])
    
    # --- Visualization Setup ---
    f = Figure(size = (1600, 600))
    
    # Axis 1: The JMAK S-Curve (Volume Fraction)
    ax1 = Axis(f[1, 1], 
               #title = "Ferrite Volume Fraction vs. Time",
               xlabel = "Time (s)", 
               ylabel = "Phase Fraction")
               
    ylims!(ax1, 0.0, 1.0)
    
    # Map the color to temp_C
    line1 = lines!(ax1, time_s, ferrite_frac, 
                   color = temp_C, colormap = :inferno, linewidth = 3)
    
    # Axis 2: Average Grain Size Over Time
    ax2 = Axis(f[1, 2], 
               #title = "Average Grain Diameter vs. Time",
               xlabel = "Time (s)", 
               ylabel = "Diameter (µm)")
               
    lines!(ax2, time_s, avg_diameter, 
           color = temp_C, colormap = :inferno, linewidth = 3)
                   
    # --- Add the Colorbar ---
    Colorbar(f[1, 3], line1, label = "Temperature (°C)")
    
    #display(f)
    
    # --- Save the High-Resolution PNG ---
    # 1. Extract the folder path from the CSV path
    out_dir = dirname(csv_path)
    
    # 2. Construct the full path for the new image
    img_path = joinpath(out_dir, "simulation_summary.png")
    
    # 3. Save the figure. px_per_unit = 2 doubles the pixel density for high res
    save(img_path, f, px_per_unit = 3)
    println("Plot successfully saved to: $img_path")
    
    return f
end

function plot_temperature_dependence(csv_path::String)
    # Read the data, separating the header row from the actual numbers
    data, header = readdlm(csv_path, ',', header=true)
    
    # Extract the columns based on the order we saved them
    time_s = Float32.(data[:, 1])
    temp_C = Float32.(data[:, 2])
    ferrite_frac = Float32.(data[:, 3])
    avg_diameter = Float32.(data[:, 5])
    
    # --- Visualization Setup ---
    # We use a taller figure to accommodate two distinct rows
    f = Figure(size = (1200, 1000))
    
    # Axis 1: Temperature vs. Time (Top row, spanning columns 1 and 2)
    ax1 = Axis(f[1, 1:2], 
               #title = "Thermal Profile",
               xlabel = "Time (s)", 
               ylabel = "Temperature (°C)")
               
    lines!(ax1, time_s, temp_C, color = :black, linewidth = 3)
    
    # Axis 2: Phase Evolution vs. Temperature (Bottom Left)
    ax2 = Axis(f[2, 1], 
               #title = "Phase Evolution vs. Temperature",
               xlabel = "Temperature (°C)", 
               ylabel = "Ferrite Area Fraction")
               
    ylims!(ax2, 0.0, 1.0)
    # Mapping color to time_s so you can see the chronological path of the curve
    lines!(ax2, temp_C, ferrite_frac, 
           color = time_s, colormap = :viridis, linewidth = 3)
    
    # Axis 3: Grain Size vs. Temperature (Bottom Right)
    ax3 = Axis(f[2, 2], 
               #title = "Grain Growth vs. Temperature",
               xlabel = "Temperature (°C)", 
               ylabel = "Average Grain Diameter (µm)")
               
    lines!(ax3, temp_C, avg_diameter, 
           color = time_s, colormap = :viridis, linewidth = 3)
           
    # Add a colorbar for time on the far right of the bottom row
    Colorbar(f[2, 3], colormap = :viridis, limits = extrema(time_s), label = "Time (s)")
    
    #display(f)
    
    # --- Save the High-Resolution PNG ---
    out_dir = dirname(csv_path)
    img_path = joinpath(out_dir, "temperature_dependence_summary.png")
    
    save(img_path, f, px_per_unit = 3)
    println("Plot successfully saved to: $img_path")
    
    return f
end

end # module kmc_gpu


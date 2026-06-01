import numpy as np
import cupy
import matplotlib.pyplot as plt
import src.ddm_analysis.ddm_functions_f as ddm
import src.flock_models.vicsek_simulation as sim  
import os

"""

Discard the first T_ss = 1000 frames to avoid the radnom initial positions transient.

"""

# simulation parameters
N = 2000
L = 2400
speed = 1.5
persistence_length = 10
# persistence_length = 16
noise_strength = np.sqrt(24 * speed / persistence_length)
Tf = 3000
T_ss = 1000

Ri_options = [0,30,35,40,45,50] 


# ddm parameters
Dt_set = [50, 100, 125, 250, 500]  # selection of episode lengths to be used
ws = [4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
tau_max = 41  # maximum tau Fourier time
taus = np.arange(1, tau_max)
K = 30  # number of Fourier directions to measure at
anglesPerDirection = (
    1  # K * anglesPerDirection angles over the interval 0 to pi are measured at,
)
# then sorted into bins containing anglesPerDirection angles each, and an average is taken of values in each bin
prominence_threshold = 0.2 # required prominence for sec function fitting - can be changed!

# runs num_simulations, outputs flockness as a function of tile size and episode length.
flocknesses_dir = "data_vicsek_2"
os.makedirs(flocknesses_dir, exist_ok=True)
num_simulations = 10

Ri_options = [0,30,35,40,45,50] 
num_simulations = 3


for Ri in Ri_options:
    for sim_id in range(num_simulations): 
        Tf = 3000
        # save density, anisos, and flocknesses.  
        file_path_A = f"{flocknesses_dir}/sim_{sim_id}_A_interaction_radius={Ri}.npy"
        file_path_D = f"{flocknesses_dir}/sim_{sim_id}_D_interaction_radius={Ri}.npy"
        file_path_F = f"{flocknesses_dir}/sim_{sim_id}_F_interaction_radius={Ri}.npy"
        if os.path.exists(file_path_A) and os.path.exists(file_path_D) and os.path.exists(file_path_F):
            print(f"sim {sim_id} already exists")
            continue
        else: 
            for paths in [file_path_A, file_path_D, file_path_F]:
                if os.path.exists(paths):
                    os.remove(paths)

        print(f"{Ri} radius, sim {sim_id} starting")

        # run the simulation
        Xs_full = sim.give_positions(speed, Ri, noise_strength, N, L, Tf)
        Xs = Xs_full[T_ss:]
        Tf = len(Xs)
        
        print(f"sim {sim_id} completed, ddm starting")

        # mm-DDM analysis
        qq_subs = ddm.give_outputpositions(0.4, K, anglesPerDirection)
        poss_Dts = ddm.give_all_splits_f(Dt_set, Xs, Tf)
        multiDDMs_q = ddm.give_AllmultiDDMs_q_f(
            Dt_set, poss_Dts, ws, qq_subs, ws[-1], L, taus
        )
        fits_peak = ddm.give_Allfits_peak_method_f(
            Dt_set, multiDDMs_q, ws, prominence_threshold
        )
        densities = ddm.box_densities(Dt_set, ws, Xs, L)
        fit_qualities = ddm.give_fit_qualities_3(Dt_set, ws, fits_peak)

        anisos = ddm.structure_consistency(fit_qualities)
        dens = ddm.structure_consistency(densities)
        # flocknesses shape = len(Dt_set), len(ws), (total tiles* total episodes sampled).
        flocknesses = ddm.parameter_product(anisos, dens, L)
        np.save(file_path_A, np.array(anisos, dtype=object))
        np.save(file_path_D, np.array(dens, dtype=object))
        np.save(file_path_F, np.array(flocknesses, dtype=object))
        print(f"{Ri} radii, sim {sim_id} A,D,F saved")
    print(f"all simulations processed for radius = {Ri}")
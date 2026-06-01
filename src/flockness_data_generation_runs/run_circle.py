import numpy as np 
import cupy
import src.ddm_analysis.ddm_functions_f as ddm
import src.flock_models.circular_isoball as sim   
import numpy as np 
import os
import matplotlib.pyplot as plt
import glob
import importlib

"""
Simulate and saves F, A, and D flockness data for an isotropic ballistic background.

"""
# Set physical parameters
L = 2400  # frame side length in pixels
v0 = 6.4  # flock speed (all flocks have the same speed in this simulation)
Ri = 250  # flock radius (all flocks have the same radius in this simulation)
Tf = 1000  # tal number of frames in simulation
N = 800
N_bg = 4200
number_of_flocks = 1

# can change to rand walk



# Set DDM measurement parameters
Dt_set = [50, 100, 125, 250, 500]  # selection of episode lengths to be used
ws = [4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]

tau_max = 41  # maximum tau Fourier time
taus = np.arange(1, tau_max)
K = 30  # number of Fourier directions to measure at
anglesPerDirection = (
    1  # K * anglesPerDirection angles over the interval 0 to pi are measured at,
)
# then sorted into bins containing anglesPerDirection angles each, and an average is taken of values in each bin
prominence_threshold = 0.1 # required prominence for sec function fitting - can be changed!

num_simulations = 1
# flocknesses_dir = "data_isoball_bg" #data save location.
# flock_numbers = [0,1,2,4,8,20]


# or can change to rand walk
D = 5
import src.flock_models.circular_rand_walk as sim  
flocknesses_dir = "data_diff_bg"





for sim_id in range(0, num_simulations): 
    # save density, anisos, and flocknesses.  
    file_path_A = f"{flocknesses_dir}/sim_{sim_id}_A_Nflocks={number_of_flocks}_iso_bg.npy"
    file_path_D = f"{flocknesses_dir}/sim_{sim_id}_D_Nflocks={number_of_flocks}_iso_bg.npy"
    file_path_F = f"{flocknesses_dir}/sim_{sim_id}_F_Nflocks={number_of_flocks}_iso_bg.npy"
    if os.path.exists(file_path_A) and os.path.exists(file_path_D) and os.path.exists(file_path_F):
        print(f"sim {sim_id} already exists")
        continue
    else: 
        for paths in [file_path_A, file_path_D, file_path_F]:
            if os.path.exists(paths):
                os.remove(paths)

    print(f"{number_of_flocks} flocks sim {sim_id} starting")

    # run the simulation
    x0 = cupy.random.uniform(Ri, L - Ri, number_of_flocks, dtype=cupy.float32)
    y0 = cupy.random.uniform(Ri, L - Ri, number_of_flocks, dtype=cupy.float32)
    Thetas = cupy.random.uniform(0, 2 * cupy.pi, number_of_flocks, dtype=cupy.float32)

    # isoball
    # Xs = sim.give_poss_hmg(N, L, v0, Tf, Thetas, x0, y0, number_of_flocks, N_bg)

    # rand walk
    Xs = sim.give_poss_hmg(N, L, v0, D, Tf, Thetas, x0, y0, number_of_flocks, N_bg)





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
    print(f"{number_of_flocks} flocks sim {sim_id} A,D,F saved")
print(f"all simulations processed for {number_of_flocks} flocks")

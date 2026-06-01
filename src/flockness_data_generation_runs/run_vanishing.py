import numpy as np 
import cupy
import src.ddm_analysis.ddm_functions_f as ddm
import src.flock_models.vanishing_isoball as sim  
import numpy as np 
import os
import matplotlib.pyplot as plt
import glob
import importlib
importlib.reload(sim)
"""
Simulate and saves F, A, and D flockness data for an isotropic ballistic background for a few T_evaps

"""

# simulation params vanishing flock

# simulation params vanishing flock
L = 2400
v0 = 3.2
Tf = 1200
num = 2
N = 200
number_of_flocks = 2
N_bg = 3200
T_evaps = [Tf, 20, 140, 260, 380, 500]
x0 = cupy.random.uniform(0, L, num, dtype=cupy.float32)
y0 = cupy.random.uniform(0, L, num, dtype=cupy.float32)
Theta = cupy.random.uniform(0, 2 * cupy.pi, num, dtype=cupy.float32)


# Set measurement parameters
Dt_set = [60, 120, 200, 240, 300, 400, 600]  # selection of episode lengths to be used
ws = [4, 6, 10, 15, 24, 40, 120]

tau_max = 41  # maximum tau Fourier time
taus = np.arange(1, tau_max)
K = 30  # number of Fourier directions to measure at
anglesPerDirection = (
    1  # K * anglesPerDirection angles over the interval 0 to pi are measured at,
)
# then sorted into bins containing anglesPerDirection angles each, and an average is taken of values in each bin
prominence_threshold = 0.1 # required prominence for sec function fitting - can be changed!


num_simulations = 5
flocknesses_dir = "evap_isoball_moreDts"
os.makedirs(flocknesses_dir, exist_ok=True)


for sim_id in range(0, num_simulations): 
    
    for T_evap in T_evaps:

        # save density, anisos, and flocknesses.  
        file_path_A = f"{flocknesses_dir}/sim_{sim_id}_A_Tevap={T_evap}.npy"
        file_path_D = f"{flocknesses_dir}/sim_{sim_id}_D_Tevap={T_evap}.npy"
        file_path_F = f"{flocknesses_dir}/sim_{sim_id}_F_Tevap={T_evap}.npy"
        if os.path.exists(file_path_A) and os.path.exists(file_path_D) and os.path.exists(file_path_F):
            print(f"sim {sim_id} already exists")
            continue
        else: 
            for paths in [file_path_A, file_path_D, file_path_F]:
                if os.path.exists(paths):
                    os.remove(paths)

        print(f"Tevap={T_evap}, sim {sim_id} starting")

        # run the simulation
        x0 = cupy.random.uniform(0, L, num, dtype=cupy.float32)
        y0 = cupy.random.uniform(0, L, num, dtype=cupy.float32)
        Theta = cupy.random.uniform(0, 2 * cupy.pi, num, dtype=cupy.float32)
        Xs = sim.give_poss_hmg(N, L, v0, T_evap, Tf, Theta, x0, y0, num=num, N_bg=N_bg)
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
        print(f"T_evap = {T_evap}, sim {sim_id} A,D,F saved")


print("varying T_evap isoball DDM finished and saved")

import multiDDMProcessing
import cupy
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu


# Returns list of episodes of length Dt, each containing time series of positions of particles
def split_frames_poss_f(Dt, poss, Tf):
    poss_split = []
    k = Tf // Dt
    for j in range(k):
        poss_split.append(poss[j * Dt : (j + 1) * Dt, :, :])
    return poss_split


# Returns list of lists of episodes of length Dt for each Dt in Dt_set
def give_all_splits_f(Dt_set, poss, Tf):
    all_splits = []
    for k in range(len(Dt_set)):
        all_splits.append(split_frames_poss_f(Dt_set[k], poss, Tf))
    return all_splits


# Returns base transforms for all positions
def give_baseTransforms(posses, outputPositions, gridSize, L):
    return multiDDMProcessing.doBaseTransform(posses, outputPositions, gridSize, L)


# stuff
# DDM is calculated chunk by chunk to reduce memory usage
def give_AllmultiDDMs_q_f(Dt_set, poss_Dts, ws, outposition_tot, gridSize, L, taus):

    all_DDMs_per_Dt = []

    for i in range(len(Dt_set)):
        DDMs_for_this_Dt = []

        for k in range(len(poss_Dts[i])):
            base_transform_chunk = give_baseTransforms(
                poss_Dts[i][k], outposition_tot, gridSize, L
            )

            DDMs_for_this_chunk = []
            for l in range(len(ws)):
                ddm_output = multiDDMProcessing.getDDMOutput(
                    base_transform_chunk, ws[l], taus
                ).get()
                DDMs_for_this_chunk.append(ddm_output)

            DDMs_for_this_Dt.append(DDMs_for_this_chunk)

            del base_transform_chunk
            cupy.get_default_memory_pool().free_all_blocks()

        all_DDMs_per_Dt.append(DDMs_for_this_Dt)

    return all_DDMs_per_Dt


def get_Allfits_peak_method(data, prominence_threshold=0.1):

    l_x, l_y, K, tau_max_minus = data.shape

    arg_maxes = np.full((l_x, l_y, K), tau_max_minus, dtype=np.float32)
    # ilx,ily,i = cell at ily,ilx and angle i, τ max_minus = max lag time possible
    for ilx in range(l_x):
        for ily in range(l_y):
            for i in range(K):
                isf_curve = data[ilx, ily, i, :]

                if not np.any(isf_curve):
                    arg_maxes[ilx, ily, i] = 0
                    continue

                peak_range = np.max(isf_curve) - np.min(isf_curve)

                if peak_range == 0:
                    continue

                required_prominence = peak_range * prominence_threshold

                peaks, irr = find_peaks(isf_curve, prominence=required_prominence)

                if peaks.size > 0:
                    arg_maxes[ilx, ily, i] = peaks[0] + 1
    return cupy.asarray(arg_maxes)


def give_Allfits_peak_method_f(Dt_set, multiDs, blockSizes, prominence_threshold=0.1):
    fit_params = []
    for i in range(len(Dt_set)):
        fit_params.append([])
        for q in range(len(multiDs[i])):
            fit_params[i].append([])
            for k in range(len(blockSizes)):
                fit_params[i][q].append(
                    get_Allfits_peak_method(multiDs[i][q][k], prominence_threshold)
                )

    return fit_params

# def recip_cos(x, a, b, c, maxim=41):
#     return np.where(
#         a / np.abs(np.cos(c * (x - b))) < maxim, a / np.abs(np.cos(c * (x - b))), maxim
#     )
 
def recip_cos(x, a, b, c, maxim=41):
    return np.where(
        a / np.abs(np.cos(c * (x - b))) < maxim, a / np.abs(np.cos(c * (x - b))), maxim
    )


def mod_sum_diffs(y_data, y_fit):
    AE = np.sum(np.abs(np.subtract(y_data, y_fit)))
    MAE = AE / np.size(y_data)
    epsilon = 1
    return 1 / (MAE + epsilon)


def give_fit_qualities(Dt_set, ws, fits):
    qualities = []
    for i in range(len(Dt_set)):
        qualities.append([])
        for q in range(len(fits[i])):
            qualities[i].append([])
            for k in range(len(ws)):
                # P(lx,ly,K)
                P = fits[i][q][k][:, :, :]
                shape = cupy.shape(P)
                K = shape[2]
                transferred_P = cupy.asnumpy(P)
                output = np.zeros((shape[0], shape[1]), dtype=np.float32)
                fourier_angles = np.linspace(0, np.pi, K, endpoint=False)

                for ilx in range(shape[0]):
                    for ily in range(shape[1]):
                        P_sub = transferred_P[ilx, ily, :]
                        min_index = (np.argmax(P_sub) + int(K / 2)) % K

                        try:
                            popt, pcov = curve_fit(
                                recip_cos,
                                fourier_angles,
                                P_sub,
                                p0=[np.amin(P_sub), np.pi * min_index / K, 1],
                            )
                        except Exception:
                            popt = [np.amin(P_sub), np.pi * min_index / K, 1]
                        quality = mod_sum_diffs(
                            P_sub,
                            recip_cos(
                                np.linspace(0, np.pi, K, endpoint=False),
                                popt[0],
                                popt[1],
                                popt[2],
                            ),
                        )

                        output[ilx, ily] = quality
                qualities[i][q].append(output)
    return qualities

def give_fit_qualities_2(Dt_set, ws, fits):
    qualities = []
    for i in range(len(Dt_set)):
        qualities.append([])
        for q in range(len(fits[i])):
            qualities[i].append([])
            for k in range(len(ws)):
                # P(lx,ly,K)
                P = fits[i][q][k][:, :, :]
                shape = cupy.shape(P)
                K = shape[2]
                transferred_P = cupy.asnumpy(P)
                output = np.zeros((shape[0], shape[1]), dtype=np.float32)
                fourier_angles = np.linspace(0, np.pi, K, endpoint=False)

                for ilx in range(shape[0]):
                    for ily in range(shape[1]):
                        P_sub = transferred_P[ilx, ily, :]
                        min_index = (np.argmax(P_sub) + int(K / 2)) % K

                        c_fixed = 1
                        sum_of_residuals = []
                        #fits a recip cosine for each 'b', only scaling 'a'
                        for b_fixed in fourier_angles: 
                            try:
                                popt, pcov = curve_fit(
                                    lambda x, a: recip_cos(x, a, b_fixed, c_fixed),
                                    fourier_angles,
                                    P_sub,
                                    p0=[np.amin(P_sub)]
                                )
                            except Exception:
                                popt = [np.amin(P_sub)]
                            
                            # sum of residuals for this b_fixed
                            y_fit = recip_cos(fourier_angles, popt[0], b_fixed, c_fixed)
                            y_data = P_sub
                            sum_of_residuals.append(np.sum(np.abs(np.subtract(y_data, y_fit))))

                        # now analyse variation in the sum of residuals (special angle?) 
                        # quality ~ 1 for no flocks, 
                        denom = np.sum(sum_of_residuals)
                        if denom == 0:
                            quality = 1
                        else:
                            quality = K * np.min(sum_of_residuals) / (denom)
                            quality = 1 / (quality + 1e-12)
                        output[ilx, ily] = quality
                qualities[i][q].append(output)
    return qualities


def give_fit_qualities_3(Dt_set, ws, fits):
    qualities = []
    for i in range(len(Dt_set)):
        qualities.append([])
        for q in range(len(fits[i])):
            qualities[i].append([])
            for k in range(len(ws)):
                # P(lx,ly,K)
                P = fits[i][q][k][:, :, :]
                shape = cupy.shape(P)
                K = shape[2]
                transferred_P = cupy.asnumpy(P)
                output = np.zeros((shape[0], shape[1]), dtype=np.float32)
                fourier_angles = np.linspace(0, np.pi, K, endpoint=False)

                for ilx in range(shape[0]):
                    for ily in range(shape[1]):
                        P_sub = transferred_P[ilx, ily, :]
                        min_index = (np.argmax(P_sub) + int(K / 2)) % K

                        c_fixed = 1
                        sum_of_residuals = []
                        #fits a recip cosine for each 'b', only scaling 'a'
                        for b_fixed in fourier_angles: 
                            try:
                                popt, pcov = curve_fit(
                                    lambda x, a: recip_cos(x, a, b_fixed, c_fixed),
                                    fourier_angles,
                                    P_sub,
                                    p0=[np.amin(P_sub)]
                                )
                            except Exception:
                                popt = [np.amin(P_sub)]
                            
                            # sum of residuals for this b_fixed
                            y_fit = recip_cos(fourier_angles, popt[0], b_fixed, c_fixed)
                            y_data = P_sub
                            sum_of_residuals.append(np.sum(np.abs(np.subtract(y_data, y_fit))))

                        # now analyse variation in the sum of residuals (special angle?) 
                        # quality ~ 0 for no flocks, ~1 for a perfect flock
                        denom = np.sum(sum_of_residuals)
                        if np.all(P_sub == 0):
                            quality = 0
                            # if no isf, arg maxes = 0 in get_Allfits_peak_method
                        elif denom == 0:
                            quality = 1
                        else:
                            quality = K * np.min(sum_of_residuals) / (denom)
                            quality = (1 - quality)
                        output[ilx, ily] = quality
                qualities[i][q].append(output)
    return qualities

def box_densities(Dt_set, ws, timeseries, L):
    Dt_set = cupy.asarray(Dt_set)
    ws = cupy.asarray(ws)
    num_frames = timeseries.shape[0]
    num_ws = len(ws)

    box_count = [ws[-1] / w for w in ws]
    densities = []

    all_box_positions = cupy.empty(
        (num_ws, num_frames, timeseries.shape[1], 2), dtype=cupy.int32
    )

    for k in range(num_ws):
        grid_size = int(box_count[k])
        all_box_positions[k] = multiDDMProcessing.assignBoxNumbers(
            timeseries, L, grid_size
        )

    for dt in Dt_set:

        episode_count = int(num_frames // dt)
        results_for_this_dt = []

        for j in range(episode_count):
            start_frame = j * dt
            end_frame = (j + 1) * dt
            results_for_this_episode = []

            for k in range(num_ws):
                grid_size = int(box_count[k])
                grid_shape = (grid_size, grid_size)
                current_positions = all_box_positions[k, start_frame:end_frame]

                flat_indices = cupy.ravel_multi_index(
                    (current_positions[..., 0], current_positions[..., 1]), grid_shape
                )
                total_counts_flat = cupy.bincount(
                    flat_indices.ravel(), minlength=grid_shape[0] * grid_shape[1]
                )
                mean_counts_grid = (total_counts_flat / dt).reshape(grid_shape)
                mean_counts_grid[mean_counts_grid < 1] = 0
                box_measure = (L * ws[k] / ws[-1]) ** 2
                density_grid = mean_counts_grid / box_measure

                results_for_this_episode.append(density_grid)

            results_for_this_dt.append(results_for_this_episode)

        densities.append(results_for_this_dt)

    return densities


def parameter_product(qualities, densities, L):
    if isinstance(qualities, list):
        return [
            parameter_product(quality, density, L)
            for quality, density in zip(qualities, densities)
        ]
    else:
        return qualities * densities * L**2


#low number cuttoff version of parameter product, sets flocknesses to 0 if too small a number of particles.
def parameter_product_lnc(qualities, densities, L, ws, min_particles=2):
    if isinstance(qualities, list):
        return [
            parameter_product_lnc(q, d, L, ws, min_particles)
            for q, d in zip(qualities, densities)
        ]
    if isinstance(qualities, cupy.ndarray):
        qualities = cupy.asnumpy(qualities)
    if isinstance(densities, cupy.ndarray):
        densities = cupy.asnumpy(densities)
    if not hasattr(qualities, "shape"):
        return np.nan
    

    #find cell size
    ws_idx = None
    for idx, w in enumerate(ws):
        grid_size = int(ws[-1] / w)
        if qualities.shape == (grid_size, grid_size):
            ws_idx = idx
            break
    if ws_idx is None:
        raise ValueError("Could not find matching ws index for qualities shape.")
    
    #remove particle counts nbelow threshold
    box_area = (L * ws[ws_idx] / ws[-1]) ** 2
    particle_count_grid = densities * box_area
    valid_mask = particle_count_grid >= min_particles

    flockness = qualities * densities * L**2
    flockness = np.nan_to_num(flockness, nan=0.0, posinf=0.0, neginf=0.0)
    flockness[~valid_mask] = 0

    return flockness

def structure_consistency(nested):
    if isinstance(nested, cupy.ndarray):
        return nested.tolist()
    if isinstance(nested, np.ndarray):
        return nested.tolist()
    if isinstance(nested, list):
        return [structure_consistency(item) for item in nested]
    return nested

# turns nested list into 1s 
def ones_like_struct(x):
    if isinstance(x, list):
        return [ones_like_struct(i) for i in x]
    else:
        return 1

def collapse_osc_f(Dt_set, aniso_fits, ws):
    a_tot = []
    for j in range(len(Dt_set)):
        w_results = []
        for w in range(len(ws)):
            flattened_values = []
            for q in range(len(aniso_fits[j])):
                list_of_lists = aniso_fits[j][q][w]
                for sublist in list_of_lists:
                    flattened_values.extend(sublist)
            w_results.append(np.array(flattened_values))
        a_tot.append(w_results)
    return a_tot




def give_histos_collapse_f(Dt_set, anisos, ws):
    return collapse_osc_f(Dt_set, anisos, ws)


def give_outputpositions(wavenumber, numberOfDirections, anglesPerDirection):
    outputAngles = cupy.linspace(
        0, cupy.pi, numberOfDirections * anglesPerDirection, endpoint=False
    )
    outputAngles = outputAngles.reshape((numberOfDirections, anglesPerDirection))
    outputPositions = cupy.zeros(
        (numberOfDirections, anglesPerDirection, 2), cupy.float32
    )
    outputPositions[:, :, 0] = (wavenumber * cupy.cos(outputAngles)).astype(
        cupy.float32
    )
    outputPositions[:, :, 1] = (wavenumber * cupy.sin(outputAngles)).astype(
        cupy.float32
    )
    return outputPositions


def binarise_metric(metric):
    binarised_metric = []
    for i in range(len(metric)):
        binarised_metric.append([])
        for j in range(len(metric[i])):
            binarised_metric[i].append([])
            for k in range(len(metric[i][j])):
                metric_array = np.asarray(metric[i][j][k])
                flat_array = metric_array.flatten()
                auto_threshold = threshold_otsu(flat_array)
                binarised_metric_array = (metric_array > auto_threshold).astype(int)
                binarised_metric[i][j].append(binarised_metric_array)
    return binarised_metric

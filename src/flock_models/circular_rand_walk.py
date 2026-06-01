# simulation functions for circular flocks (ballistic + some noise) with a background gas of diffusive particles (random walk with a diffusion coefficient D)
# e.g. Xs = give_poss_hmg(N, L, v0, D, Tf, Thetas, x0, y0, num=number_of_flocks, N_bg=N_bg) generates the simulation data

# Generate a circular flock of randomly located particles with the specified number of particles, radius, and starting position
import numpy as np
import cupy
import random
import matplotlib.pyplot as plt
import copy
import ddm_functions_f as ddm
from scipy.optimize import curve_fit
from scipy import special
from scipy import integrate

def circle(N, ri, x0, y0):
    r0 = cupy.array([x0, y0], dtype=cupy.float32)
    r = ri * cupy.sqrt(cupy.random.uniform(0, 1, size=(N)))
    phi = cupy.random.uniform(0, 2 * cupy.pi, size=(N), dtype=cupy.float32)

    pos = cupy.zeros((N, 2), dtype=cupy.float32)
    pos[:, 0] = r * cupy.cos(phi)
    pos[:, 1] = r * cupy.sin(phi)

    return r0 + pos

# sets up random initial particle positions
def ini_hmg(N, L, ri, x0, y0, num=1, N_bg=0): 
    pos = cupy.zeros((N * num + N_bg, 2), dtype=cupy.float32)


    for j in range(num):
        pos[j * N : (j + 1) * N] = circle(N, ri, x0=x0[j], y0=y0[j])

    flock_centres = cupy.stack([x0, y0], axis=1)
    bg = []

    bg_left = N_bg
    while bg_left > 0:
        candidates = cupy.random.uniform(0, L, size=(bg_left, 2)).astype(cupy.float32)
        # delta[i,j,k] is the k-th component of the vector from the j-th flock centre to the i-th background particle
        # shape = (N_bg, num, 2)
        delta = candidates[:, None, :] - flock_centres[None, :, :]
        delta = delta - L * cupy.round(delta / L)  
        distances2 = cupy.sum(delta**2, axis=2) 
        valid = cupy.all(distances2 > ri**2, axis = 1)
        accepted = candidates[valid]
        bg.append(accepted)
        bg_left -= accepted.shape[0]
    background_particles = cupy.concatenate(bg, axis=0)[:N_bg]
    pos[N * num:] = background_particles
    return pos
    

def step_fwd_hmg(
    pos,
    v0,
    D,
    Theta,
    N,
    L,
    flock_x,
    flock_y,
    num=1,
    N_bg=0,
):
    # equal flock and bg density
    density = (N * num + N_bg) / L**2
    radius_hmg = cupy.min(cupy.array([(N / (cupy.pi * density)) ** (1 / 2), L / 2]))

    pos_new = cupy.zeros(cupy.shape(pos))
    Theta = cupy.array(Theta, dtype=cupy.float32)

    # update the flocks
    for j in range(num):
        # phi = cupy.random.uniform(0, 2 * cupy.pi, size=N, dtype=cupy.float32)
        # particles in a flock advance by v0 in the specified direction, noise removed.
        pos_new[j * N : (j + 1) * N, 0] = (
            pos[j * N : (j + 1) * N, 0]
            + v0 * cupy.cos(Theta[j])   
        )
        pos_new[j * N : (j + 1) * N, 1] = (
            pos[j * N : (j + 1) * N, 1]
            + v0 * cupy.sin(Theta[j])     
        )
    pos_new[: N * num] = pos_new[: N * num] % L
    

    # background particles move stochastically with diffusion coefficient D,
    noise = cupy.random.normal(0, 1, size = (N_bg, 2))
    bg = pos[N*num:] + cupy.sqrt(2*D)*noise
    bg %= L



    # check bg overlaps with flocks
    # if multiple flock overlaps, correct the first one and re check validity. 
    # if more than 100 attempts to correct, give up and accept the position
    flock_centres = cupy.stack([flock_x, flock_y], axis=1).astype(cupy.float32)
    check_counter = 0
    
    # split into overlapping and non-overlapping particles
    delta = bg[:, None, :] - flock_centres[None, :, :]
    delta = delta - L * cupy.round(delta / L)
    distances2 = cupy.sum(delta**2, axis=2)
    active = cupy.any(distances2 < radius_hmg**2, axis=1)
    active_idx = cupy.where(active)[0]


    flock_dirs = cupy.stack([cupy.cos(Theta), cupy.sin(Theta)], axis=1)
    while check_counter < 1000:

        if active_idx.size == 0:
            break
        
        delta = bg[active_idx, None, :] - flock_centres[None, :, :]
        delta = delta - L * cupy.round(delta / L)
        distances2 = cupy.sum(delta**2, axis=2)
        overlaps = distances2 < radius_hmg**2

        if not cupy.any(overlaps):
            break

        # invalid[i] is True if the i-th active particle overlaps with any flock, and False otherwise
        # the ith actice particle is the active_idx[i]th particle in the original bg array 
        invalid = cupy.any(overlaps, axis=1)
        idx_local = cupy.where(invalid)[0]
        idx_global = active_idx[idx_local]
        flock_idx = cupy.argmax(overlaps[idx_local], axis=1)


        # vectors flock centre to particle for overlaps
        vector = -delta[idx_local, flock_idx]
        norm = cupy.sqrt(cupy.sum(vector**2, axis=1))
        reciprocal_norm = cupy.where(norm == 0, 0, 1/norm)

        # flips all overlapping particles across the flock centre and adds a component in the direction of the flip:
        # bg[idx_global] += 2 * vector + v0 * vector * reciprocal_norm[:, None]
        bg[idx_global] += 2*vector + v0 * vector * reciprocal_norm[:, None]
        bg[idx_global] %= L

        # recheck which particles are still overlapping after the correction
        delta = bg[active_idx, None, :] - flock_centres[None, :, :]
        delta = delta - L * cupy.round(delta / L)
        distances2 = cupy.sum(delta**2, axis=2)
        active_mask = cupy.any(distances2 < radius_hmg**2, axis=1)
        active_idx = active_idx[active_mask]

        check_counter += 1

    pos_new[N*num:] = bg
    return pos_new


def give_poss_hmg(N, L, v0, D, Tf, Theta, x0, y0, num=1, N_bg=1):

    density = (N * num + N_bg) / L**2
    radius_hmg = cupy.min(cupy.array([(N / (cupy.pi * density)) ** (1 / 2), L / 2]))

    flocks_x = copy.deepcopy(x0)
    flocks_y = copy.deepcopy(y0)

    poss = cupy.zeros((Tf, N * num + N_bg, 2), dtype=cupy.float32)
    poss_initialised = ini_hmg(N, L, radius_hmg, x0=x0, y0=y0, num=num, N_bg=N_bg)

    

    poss[0] = step_fwd_hmg(
        poss_initialised,
        v0,
        D,
        Theta,
        N,
        L,
        x0,
        y0,
        num=num,
        N_bg=N_bg,
    )
    for frame in range(1, Tf):
        flocks_x = (flocks_x + v0 * cupy.cos(Theta)) % L
        flocks_y = (flocks_y + v0 * cupy.sin(Theta)) % L
        poss[frame] = step_fwd_hmg(
            poss[frame - 1],
            v0,
            D,
            Theta,
            N,
            L,
            flocks_x,
            flocks_y,
            num=num,
            N_bg=N_bg,
        )
    return poss
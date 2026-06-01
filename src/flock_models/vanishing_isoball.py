import numpy as np
import cupy
import matplotlib.pyplot as plt
import copy




"""
vanishing flock:
	1. Trigger flock death
	2. All flock particles assigned isoball dynamics
New flock centre found. N closest particles become the flock. Sort these to the front of the positions array (flock indices)

can run for mutiple flocks.

N = number of particles
L = x and y extents of the simulation grid
v0 = speed
Tf = full simultion time
num = number of flocks
N_bg = numbe of particles in the background gas
T_evap = time flocks exist for before evaporating into the backgroudn and a new flock is made at a new location.


Initial positions and headings:
x0 = cupy.random.uniform(Ri, L - Ri, number_of_flocks, dtype=cupy.float32)
y0 = cupy.random.uniform(Ri, L - Ri, number_of_flocks, dtype=cupy.float32)
Thetas = cupy.random.uniform(0, 2 * cupy.pi, number_of_flocks, dtype=cupy.float32)
Xs = give_poss_hmg(N, L, v0, T_evap, Tf, Thetas, x0, y0, num=number_of_flocks, N_bg=N_bg) #gives the particle position time series


"""

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
    Theta,
    N,
    L,
    flock_x,
    flock_y,
    num=1,
    v_bgs=cupy.asarray([]),
    Theta_bgs=cupy.asarray([]),
    N_bg=0,
):
    # equal flock and bg density
    density = (N * num + N_bg) / L**2
    radius_hmg = cupy.min(cupy.array([(N / (cupy.pi * density)) ** (1 / 2), L / 2]))

    pos_new = cupy.zeros(cupy.shape(pos))
    Theta = cupy.array(Theta, dtype=cupy.float32)

    # update the flocks
    for j in range(num):
        pos_new[j * N : (j + 1) * N, 0] = (
            pos[j * N : (j + 1) * N, 0]
            + v0 * cupy.cos(Theta[j])   
        )
        pos_new[j * N : (j + 1) * N, 1] = (
            pos[j * N : (j + 1) * N, 1]
            + v0 * cupy.sin(Theta[j])     
        )
    pos_new[: N * num] = pos_new[: N * num] % L
    

    # particles in the background gas advance at a constant velocity (that is different for each particle),
    pos_new[num * N:, 0] = pos[num * N:, 0] + v_bgs * cupy.cos(
        Theta_bgs
    )
    pos_new[num * N:, 1] = pos[num * N:, 1] + v_bgs * cupy.sin(
        Theta_bgs
    )
    pos_new[num * N:] = pos_new[num * N:] % L
    bg = pos_new[num*N:]



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

def evap_flocks(poss, v_bgs, Theta_bgs, v0, N, num, L, flocks_x, flocks_y):
    # flocks --> isoball
    N_bg = len(poss) - N*num
    density = (N * num + N_bg) / L**2
    radius_hmg = cupy.min(cupy.array([(N / (cupy.pi * density)) ** (1 / 2), L / 2]))
    old_flock_centres = cupy.stack((flocks_x, flocks_y), axis = 1)

    # find new flock positions, larger than 2r_flock from the old flock centres, AND 2r from the new flock centres
    new_flock_centres = cupy.zeros((num,2))
    accepted_count = 0
    safe_dist2 = (2*radius_hmg)**2

    while accepted_count < num:
        candidate = cupy.random.uniform(0, L, size = (1,2))

        # check against old centres
        delta_old = candidate - old_flock_centres
        delta_old = delta_old - L * cupy.round(delta_old / L) 
        d_old2 = cupy.sum(delta_old**2, axis=1) 
        valid4_old = cupy.all(d_old2 > safe_dist2)

        # check against new centres
        if accepted_count > 0:
            delta_new = candidate - new_flock_centres[:accepted_count]
            delta_new = delta_new - L * cupy.round(delta_new / L) 
            d_new2 = cupy.sum(delta_new**2, axis=1) 
            valid4_new = cupy.all(d_new2 > safe_dist2)
        else:
            valid4_new = True

        if valid4_new and valid4_old:
            new_flock_centres[accepted_count] = candidate[0]
            accepted_count += 1
    # extend the vbgs and theta bgs arrays to include flock parrticles so that bg particles keep their same speeds and directions when poss is resorted.
    new_theta_bg = cupy.random.uniform(0, 2*cupy.pi, size = N*num)
    new_vs_bg = cupy.random.uniform(0, 2*v0, size = N*num)
    vs_all = cupy.concatenate((new_vs_bg, v_bgs), axis = 0)
    thetas_all = cupy.concatenate((new_theta_bg, Theta_bgs), axis = 0)

    # find the N nearest particles to each flock, move these to the front of poss array. 
    bg = poss.copy()
    flock_pos = []
    

    for i in range(num):
        delta = bg - new_flock_centres[i]
        delta = delta - L * cupy.round(delta / L)
        dist2 = cupy.sum(delta**2, axis =1)

        sorted_idx = cupy.argsort(dist2)
        # sort vs thetas and poss by nearest distacnce
        vs_all = vs_all[sorted_idx]
        thetas_all = thetas_all[sorted_idx]
        bg = bg[sorted_idx]

        # add to flock poss and cut flock pos off bg 
        vs_all = vs_all[N:]
        thetas_all = thetas_all[N:]
        flock_pos.append(bg[:N])
        bg = bg[N:]

    poss_new = cupy.concatenate((*flock_pos, bg), axis = 0)

    # vs_all and thetas_all are the new vbgs and thetabgs
    return poss_new, new_flock_centres, vs_all, thetas_all

def give_poss_hmg(N, L, v0, T_evap, Tf, Theta, x0, y0, num=1, N_bg=1):

    Theta = cupy.atleast_1d(cupy.array(Theta, dtype=cupy.float32))

    density = (N * num + N_bg) / L**2
    radius_hmg = cupy.min(cupy.array([(N / (cupy.pi * density)) ** (1 / 2), L / 2]))

    flocks_x = copy.deepcopy(x0)
    flocks_y = copy.deepcopy(y0)

    poss = cupy.zeros((Tf, N * num + N_bg, 2), dtype=cupy.float32)
    poss_initialised = ini_hmg(N, L, radius_hmg, x0=x0, y0=y0, num=num, N_bg=N_bg)
    v_bgs = cupy.zeros(N_bg)
    Theta_bgs = cupy.zeros(N_bg)
    for i in range(N_bg):
        v_bgs[i] = cupy.random.uniform(0, 2 * v0)
        Theta_bgs[i] = cupy.random.uniform(0, 2 * cupy.pi)
    poss[0] = step_fwd_hmg(
        poss_initialised,
        v0,
        Theta,
        N,
        L,
        x0,
        y0,
        num=num,
        v_bgs=v_bgs,
        Theta_bgs=Theta_bgs,
        N_bg=N_bg,
    )
    for frame in range(1, Tf):
        flocks_x = (flocks_x + v0 * cupy.cos(Theta)) % L
        flocks_y = (flocks_y + v0 * cupy.sin(Theta)) % L
        poss[frame] = step_fwd_hmg(
            poss[frame - 1],
            v0,
            Theta,
            N,
            L,
            flocks_x,
            flocks_y,
            num=num,
            v_bgs=v_bgs,
            Theta_bgs=Theta_bgs,
            N_bg=N_bg,
        )  
        if frame % T_evap == 0:
            poss_new, new_flock_centres, Theta_bgs, v_bgs = evap_flocks(poss[frame], v_bgs, Theta_bgs, v0, N, num, L, flocks_x, flocks_y)
            new_flock_Thetas = cupy.random.uniform(0, 2 * cupy.pi, num, dtype=cupy.float32)
            
            # update sim params after flock death and regen
            Theta = cupy.atleast_1d(new_flock_Thetas)
            flocks_x = new_flock_centres[:, 0]
            flocks_y = new_flock_centres[:, 1]
            poss[frame] = poss_new

    return poss
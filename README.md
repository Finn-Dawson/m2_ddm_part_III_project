# m2_ddm_part_III_project

This repository contains the computation pipeline developed to apply m2-DDM to numerical 'forward' simulations of particles showing flocking behaviour, developed as part of a part-III Physics project. 

## Repository layout
- flockmodels/
    - vicsek_simulation.py : neighbour alignment interactions with interaction form specified.
    - circular_isoball.py : Circular flocks on an isotropic ballistic background, constant density model.
    - circular_rand_walk.py : Circular flocks on an radnom walk background, constant density model.
    - vanishing_isoball.py : Circular flocks that vanish and reappear after T_{evap}.
- ddm-analysis/
    - ddm_functions_f.py : neighbour alignment interactions with interaction form specified.
    - multiDDMProcessing.py : Circular flocks on an isotropic ballistic background, constant density model.
- flockness-data-generation-runs/ :
    - sim_run_vicsek.py : neighbour alignment interactions with interaction form specified.
    - circular_isoball.py : Circular flocks on an isotropic ballistic background, constant density model.
    - circular_rand_walk.py : Circular flocks on an radnom walk background, constant density model.
    - vanishing_isoball.py : Circular flocks that vanish and reappear after T_{evap}.

## Requirements
Both the simulations and the DDM analysis rely on GPU acceleratio, ensure you have a CUDA environment configured.

Standard Python dependencies install via `pip`:
`pip`:
```bash
pip install numpy cupy scipy matplotlib




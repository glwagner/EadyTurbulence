#!/bin/bash

#SBATCH --job-name=eady
#SBATCH --output=slurm-eady-%j.out
#SBATCH --error=slurm-eady-%j.err
#SBATCH --time=12:00:00
#SBARCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres="gpu:4" # GPUs per Node
#SBATCH --cpus-per-task=4

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

module load spack/0.1
module load gcc/8.3.0 # to get libquadmath
module load julia/1.4.1
module load cuda/10.1.243
module load openmpi/3.1.4-pmi-cuda
module load py-matplotlib/3.1.1

cd /home/glwagner/EadyTurbulence/Oceananigans

# CUDA_VISIBLE_DEVICES=0 unbuffer julia --color=yes --project three_layer_constant_fluxes.jl -Nh 256 -Nz 256 --buoyancy-flux 1e-8 --momentum-flux 0

unbuffer julia --color=yes --project run_small_eady_problem.jl

unbuffer julia --color=yes --project plot_small_eady_problem.jl

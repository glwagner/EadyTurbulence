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

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#cd $DIR

CUDA_VISIBLE_DEVICES=0 unbuffer julia --project eady_surface_waves.jl --Nh 192 --Nz 96 --years 1 --geostrophic-shear 1.0 --surface-wave-amplitude 0.0 2>&1 | tee no_waves_a10.out &
CUDA_VISIBLE_DEVICES=1 unbuffer julia --project eady_surface_waves.jl --Nh 192 --Nz 96 --years 1 --geostrophic-shear 1.0 --surface-wave-amplitude 1.0 2>&1 | tee waves_1_a10.out &
CUDA_VISIBLE_DEVICES=2 unbuffer julia --project eady_surface_waves.jl --Nh 192 --Nz 96 --years 1 --geostrophic-shear 0.5 --surface-wave-amplitude 0.0 2>&1 | tee no_waves_a05.out &
CUDA_VISIBLE_DEVICES=3 unbuffer julia --project eady_surface_waves.jl --Nh 192 --Nz 96 --years 1 --geostrophic-shear 0.5 --surface-wave-amplitude 1.0 2>&1 | tee waves_1_a05.out &

sleep 42480 # sleep for 11.8 hours

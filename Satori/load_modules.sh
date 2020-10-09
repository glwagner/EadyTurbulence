#!/bin/bash

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

module load spack/0.1
module load gcc/8.3.0 # to get libquadmath
module load py-matplotlib/3.1.1
module load cuda/10.1.243
module load openmpi/3.1.4-pmi-cuda
module load julia/1.4.1

# EadyTurbulence

This repository contains scripts for simulating the baroclinic equilibration of the "Eady problem".

## Notes

* To download the code in this repository, open a terminal and type 

```
$ git clone https://github.com/glwagner/EadyTurbulence.git
```

* The scripts in this repository currently require the `glw/eady-example` branch of `Oceananigans.jl`.

* These scripts also rely on `PyPlot`, `Random`, and `Printf`.

* To use these packages within the julia 'environment' provided in `Project.toml`, do:

```
$ cd EadyTurbulence # navigate to this repository's base directory
$ julia --project # open julia with the `EadyTurbulence` environment activated
```

Alternatively, the packages can be installed by opening julia and running

```
julia>] # "close backet" enters package manager mode
pkg> add PyPlot Random Printf https://github.com/climate-machine/Oceananigans.jl.git#glw/eady-example
```

* To run the examples in this repository, "include" them at the REPL:

```julia
julia> include("simple_eady.jl")
```

* The script `simple_eady.jl` sets up an Eady equilibration example with simple boundary conditions.
* The script `eady_turbulence.jl` sets up an Eady equilibration example allowing for various boundary conditions defined by functions in the auxiliary script `eady_utils.jl`.
* The script `eady_utils.jl` defines functions that help users set up linear drag, quadratic drag, and Monin-Obukhov boundary conditions.

## Some challenges

* Identifying the magnitude of the horizontal biharmonic and vertical Laplacian diffusivities is tricky and may depend on the prescribed ambient shear / baroclinic growth rate

* Choosing the magnitude of noise in the initial condition is also challenging. Some noise seems to be beneficial for numerical stability / transition to turbulence, but too much causes obvious issues.

# EadyTurbulence

This repository contains scripts for simulating the baroclinic equilibration of the "Eady problem".

* To download the code in this repository, open a terminal and type 

```
$ git clone https://github.com/glwagner/EadyTurbulence.git
```

## Notes on running the Oceananigans scripts

* Julia v1.4 is required to run the Oceananigans scripts.

* To install the julia packages that are needed to run the Oceananigans scripts:

    - Open a terminal and navigate to the Oceananigans directory:

    ```bash
    cd EadyTurbulence/Oceananigans
    ```

    - Open julia:

    ```bash
    julia --project
    ```
    
    - Instantiate packages listed in `Project.toml`:
    
    ```julia
    using Pkg
    Pkg.instantiate()
    ```

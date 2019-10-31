using Oceananigans, Random, Printf

# Some useful (essential) utilities:
include("eady_utils.jl")

#####
##### Parameters
#####

# Resolution
Nh = 128                 # horizontal resolution
Nz = 32                  # vertical resolution

# Domain size            
Lh = 1000e3              # [meters] horizontal domain extent
Lz = 1000                # [meters] vertical domain extent
Δh = Lh / Nh             # [meters] horizontal grid spacing for diffusivity calculations
Δz = Lz / Nz             # [meters] vertical grid spacing for diffusivity calculations

# Physical parameters
  f = 1e-4               # [s⁻¹] Coriolis parameter
 N² = 1e-5               # [s⁻²] Initial buoyancy gradient 
  α = 1e-4               # [s⁻¹] background shear

 κh = Δh^2 / 2day        # [m² s⁻¹] Laplacian horizontal diffusivity
κ₄h = Δh^4 / 2day        # [m⁴ s⁻¹] Biharmonic horizontal diffusivity

 κv = (Δz / Δh)^2 * κh   # [m² s⁻¹] Laplacian vertical diffusivity
κ₄v = (Δz / Δh)^4 * κ₄h  # [m⁴ s⁻¹] Biharmonic vertical diffusivity

end_time = 60day # Simulation end time

print_eady_parameters(N², f, α, Lz) # prints useful info about the Eady problem

# These functions define various physical boundary conditions, 
# and are defined in the file eady_utils.jl

#####
##### Choose boundary conditions and the turbulence closure
#####

ubcs, vbcs, bbcs, bc_parameters = linear_drag_boundary_conditions(N²=N², μ=1/30day, H=Lz)
#ubcs, vbcs, bbcs, bc_parameters = quadratic_drag_boundary_conditions(N²=N², Cd=0.002)
#ubcs, vbcs, bbcs, bc_parameters = Monin_Obukhov_boundary_conditions(N²=N², roughness_length=1.0, Von_Karman_constant=0.41)

# Get forcing functions and parameters for a linear geostrophic flow ψ = -α y z, where
# α is the geostrophic shear and horizontal buoyancy gradient.
forcing, forcing_parameters = background_geostrophic_flow_forcing(geostrophic_shear=α, f=f)

# Turbulence closure: 
#closure = ConstantAnisotropicDiffusivity(νh=κh, κh=κh, νv=κv, κv=κv)
#closure = AnisotropicMinimumDissipation()
closure = HorizontalBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h, νv=κv, κv=κv)

# Form a prefix from chosen resolution, boundary condition, and closure name
output_filename_prefix = string("eady_turb_Nh", Nh, "_Nz", Nz, "_", 
                                bottom_bc_name(ubcs), "_", closure_name(closure))

println("Outputing to file: $output_filename_prefix")

#####
##### Instantiate the model
#####

# Model instantiation
model = Model( grid = RegularCartesianGrid(size=(Nh, Nh, Nz), halo=(2, 2, 2), x=(-Lh/2, Lh/2), y=(-Lh/2, Lh/2), z=(-Lz, 0)),
       architecture = CPU(),
           coriolis = FPlane(f=f),
           buoyancy = BuoyancyTracer(), tracers = :b,
            forcing = forcing,
            closure = closure,
boundary_conditions = BoundaryConditions(u=ubcs, v=vbcs, b=bbcs),
# "parameters" is a NamedTuple of user-defined parameters that can be used in boundary condition and forcing functions.
         parameters = merge(bc_parameters, forcing_parameters))

#####
##### Set initial conditions
#####

# A noise function, damped at the boundaries
Ξ(z) = rand() * z * (z + model.grid.Lz)

# Buoyancy: linear stratification plus noise
b₀(x, y, z) = N² * z + 1e-9 * Ξ(z) * α * f * model.grid.Ly

# Velocity: noise
u₀(x, y, z) = 1e-9 * Ξ(z) * α * model.grid.Lz

set!(model, u=u₀, v=u₀, b=b₀)

#####
##### Set up diagnostics and output
#####

# Diagnostics that return the maximum absolute value of `u, v, w` by calling
# `umax(model), vmax(model), wmax(model)`:
umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

# Set up output. Here we output the velocity and buoyancy fields at intervals of one day.
fields_to_output = merge(model.velocities, (b=model.tracers.b,))
output_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); 
                                 interval=day, prefix=output_filename_prefix,
                                 force=true, max_filesize=10GiB)

# The TimeStepWizard manages the time-step adaptively, keeping the CFL close to a
# desired value.
wizard = TimeStepWizard(cfl=0.05, Δt=20.0, max_change=1.1, max_Δt=20minute)

#####
##### Time step the model forward
#####

using PyPlot
fig, axs = subplots(ncols=2)

# This time-stepping loop runs until end_time is reached. It prints a progress statement
# every 100 iterations.
while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = Base.@elapsed time_step!(model, 10, wizard.Δt)

    #if model.clock.iteration % 100 == 0
        ## Print a progress message
        @printf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
                umax(model), vmax(model), wmax(model), prettytime(walltime))
    #end

    if model.clock.iteration % 100 == 0
        sca(axs[1]); cla()
        imshow(interior(model.velocities.u)[:, :, Nz])

        sca(axs[2]); cla()
        imshow(interior(model.velocities.w)[:, :, Nz-1])
        pause(1.0)
    end
end

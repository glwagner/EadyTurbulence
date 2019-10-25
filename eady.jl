using Oceananigans, PyPlot, Random, Printf

using Oceananigans.TurbulenceClosures: ∂x_faa, ∂x_caa, ▶x_faa, ▶y_aca, ▶x_caa, ▶xz_fac

# Resolution
Nx = 64
Ny = 64
Nz = 16 

# Domain size
Lx = 100e3
Ly = 100e3
Lz = 200

# Physical parameters
 f = 1e-4     # Coriolis parameter
N² = 1e-6     # Stratification in the "halocline"
 μ = 1e-6     # [s⁻¹] drag coefficient
 α = 1e-3     # [s⁻¹] background shear

# Simulation end time
end_time = 30day

#####
##### Boundary conditions
#####

@inline x_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * grid.Lz * U.u[i, j, 1]
@inline y_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * grid.Lz * U.v[i, j, 1]

u_bcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, x_linear_drag))
v_bcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, y_linear_drag))

#####
##### "Forcing" terms associated with background balanced flow
#####

# - α z ∂ₓu x̂
u_forcing(i, j, k, grid, time, U, C, p) = @inbounds (
    - p.α * ▶xz_fac(i, j, k, grid, U.w)
    - p.α * grid.zC[k] * ∂x_faa(i, j, k, grid, ▶x_caa, U.u) )

# - α z ∂ₓv ŷ - α z ∂ₓw ẑ  
v_forcing(i, j, k, grid, time, U, C, p) = @inbounds p.α * grid.zC[k] * ∂x_caa(i, j, k, grid, ▶x_faa, U.v)
w_forcing(i, j, k, grid, time, U, C, p) = @inbounds p.α * grid.zF[k] * ∂x_caa(i, j, k, grid, ▶x_faa, U.w)

# - α z ∂ₓb + α f v
b_forcing(i, j, k, grid, time, U, C, p) = @inbounds (
        p.f * p.α * ▶y_aca(i, j, k, grid, U.v)
      - p.α * grid.zC[k] * ∂x_caa(i, j, k, grid, ▶x_faa, C.b))

#####
##### Model instantiation
#####

model = Model(
         architecture = CPU(),
                 grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(-Lz, 0)),
             coriolis = FPlane(f=f),
             buoyancy = BuoyancyTracer(),
              tracers = :b,
              closure = ConstantAnisotropicDiffusivity(νv=1e-2, νh=10, κv=1e-2, κh=10),
              forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing),
  boundary_conditions = BoundaryConditions(u=u_bcs, v=v_bcs),
           parameters = (α=α, f=f, μ=μ)
)

# Initial condition
Ξ(z) = rand() * z * (z + model.grid.Lz)

b₀(x, y, z) = N² * z + 1e-6 * Ξ(z) * α * f * model.grid.Ly
u₀(x, y, z) = 1e-6 * Ξ(z) * α * model.grid.Lz

set!(model, u=u₀, v=u₀, b=b₀)

# A diagnostic that returns the maximum absolute value of `w` by calling
# `wmax(model)`:

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

# Set up output
fields_to_output = merge(model.velocities, (b=model.tracers.b,))
output_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=day, prefix="beaufort_gyre",
                                 force=true)

# Create a figure
close("all")
fig, axs = subplots(ncols=3, figsize=(16, 6))

function makeplot!(fig, axs, model)

    fig.suptitle("\$ t = \$ $(prettytime(model.clock.time))")

    sca(axs[1]); cla()
    title("\$ u(x, y, z=0) \$")
    imshow(interior(model.velocities.u)[:, :, Nz])

    sca(axs[2]); cla()
    title("\$ v(x, y, z=0) \$")
    imshow(interior(model.velocities.v)[:, :, Nz])

    kplot = Nz - 2
    sca(axs[3]); cla()
    title("\$ w(x, y, z=$(model.grid.zF[kplot]))")
    imshow(interior(model.velocities.w)[:, :, kplot])
    
    [ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false) for ax in axs]

    return nothing
end

wizard = TimeStepWizard(cfl=0.01, Δt=20.0, max_change=1.1, max_Δt=10minute)

while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = @elapsed time_step!(model, 10, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            umax(model), vmax(model), wmax(model), prettytime(walltime))

    if model.clock.iteration % 1000 == 0
        model.architecture == CPU() && makeplot!(fig, axs, model)
    end
end

using Pkg

Pkg.instantiate()

using JLD2
using Plots
using Printf
using Statistics
using ArgParse

using Oceananigans
using Oceananigans.Grids
using Oceananigans.AbstractOperations
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.OutputWriters
using Oceananigans.Advection

using Oceananigans.Grids: x_domain, y_domain, z_domain # for nice domain limits
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.SurfaceWaves: UniformStokesDrift

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 32
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 32
            arg_type = Int

        "--geostrophic-shear"
            help = """The geostrophic shear non-dimensionalized by f."""
            default = 1.0
            arg_type = Float64

        "--years"
            help = """The length of the simulation in years."""
            default = 1.0
            arg_type = Float64

        "--surface-wave-amplitude"
            help = """Amplitude of the surface wave field."""
            default = 0.0
            arg_type = Float64
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

Nh = args["Nh"]
Nz = args["Nz"]
α_f = args["geostrophic-shear"]
stop_years = args["years"]
year = 365day

# Surface waves
a = args["surface-wave-amplitude"]
ω = 2π / 16

const k = ω^2 / 9.81
const uˢ = a^2 * k * ω

∂z_uˢ(z, t) = 1 / 2k * uˢ * exp(2k * z)
surface_waves = uˢ == 0 ? nothing : UniformStokesDrift(∂z_uˢ=∂z_uˢ)

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 1e6), y=(0, 1e6), z=(-1e3, 0))

prefix = @sprintf("eady_surface_waves_uS%.3f_Nh%d_Nz%d_αf%.1e", uˢ, grid.Nx, grid.Nz, α_f)

coriolis = FPlane(f=1e-4) # [s⁻¹]

N² = 1e-5 # s⁻¹, buoyancy frequency
                            
background_parameters = (  α = α_f * coriolis.f, # s⁻¹, geostrophic shear
                           f = coriolis.f,        # s⁻¹, Coriolis parameter
                          Lz = grid.Lz)           # m, ocean depth

## Background fields are defined via functions of x, y, z, t, and optional parameters
U(x, y, z, t, p) = + p.α * (z + p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y

U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

drag_coefficient = 1e-3

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)
    
drag_bc_u = BoundaryCondition(Flux, bottom_drag_u, field_dependencies=(:u, :v), parameters=drag_coefficient)
drag_bc_v = BoundaryCondition(Flux, bottom_drag_v, field_dependencies=(:u, :v), parameters=drag_coefficient)

b_bcs = TracerBoundaryConditions(grid, bottom=GradientBoundaryCondition(N²), top=GradientBoundaryCondition(N²))

u_bcs = UVelocityBoundaryConditions(grid, bottom = drag_bc_u) 
v_bcs = VVelocityBoundaryConditions(grid, bottom = drag_bc_v)

κ₂z = 1e-4 # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e8 # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

Laplacian_vertical_diffusivity = AnisotropicDiffusivity(νh=0, κh=0, νz=κ₂z, κz=κ₂z)
biharmonic_horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

# # Model instantiation

model = IncompressibleModel(
           architecture = GPU(),
                   grid = grid,
              advection = WENO5(),
            timestepper = :RungeKutta3,
               coriolis = coriolis,
                tracers = :b,
               buoyancy = BuoyancyTracer(),
      background_fields = (b=B_field, u=U_field),
                closure = (Laplacian_vertical_diffusivity, biharmonic_horizontal_diffusivity),
          surface_waves = surface_waves,
    boundary_conditions = (u=u_bcs, v=v_bcs)
)

## A noise function, damped at the top and bottom
Ξ(z) = randn() * z/grid.Lz * (z/grid.Lz + 1)

## Scales for the initial velocity and buoyancy
Ũ = 1e-9 * background_parameters.α * grid.Lz
B̃ = 1e-9 * background_parameters.α * coriolis.f * grid.Ly

uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = N² * z + B̃ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

Ū = sum(model.velocities.u.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
V̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)

model.velocities.u.data.parent .-= Ū
model.velocities.v.data.parent .-= V̄

## Calculate absolute limit on time-step using diffusivities and 
## background velocity.
Ū = background_parameters.α * grid.Lz

cfl = 1.0
cfl = cfl * min(cfl, drag_coefficient * grid.Δx / grid.Δz)

abs_max_Δt = 20minutes

max_Δt = min(abs_max_Δt, cfl * grid.Δx / Ū, 0.5 * grid.Δx^4 / κ₄h, 0.5 * grid.Δz^2 / κ₂z)

wizard = TimeStepWizard(cfl=cfl, Δt=0.01*max_Δt, max_change=1.1, max_Δt=max_Δt)

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

mutable struct ProgressMessage{T}
    wall_time :: T
end

progress = ProgressMessage(time_ns())

function (p::ProgressMessage)(sim)

    @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - p.wall_time)),
            prettytime(sim.Δt.Δt),
            CFL(sim.model))

    p.wall_time = time_ns()

    return nothing
end

simulation = Simulation(model, Δt = wizard, iteration_interval = 10,
                                                     stop_time = stop_years * year,
                                                      progress = progress)

push!(simulation.stop_criteria, sim -> isnan(sim.model.clock.time))

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u)) # Vertical vorticity [s⁻¹]
δ = ComputedField(-∂z(w)) # Horizontal divergence, or ∂x(u) + ∂y(v) [s⁻¹]

## Eddy kinetic energy and buoyancy flux
f = coriolis.f

eddy_kinetic_energy = @at (Cell, Cell, Cell)  (u^2 + v^2 + w^2) / 2  
buoyancy_flux = @at (Cell, Cell, Cell)  v * b

 e = ComputedField(eddy_kinetic_energy)
vb = ComputedField(buoyancy_flux)
b² = ComputedField(b^2)
ζ² = ComputedField(ζ^2)

horizontal_average_u  = mean(u,     dims=(1, 2))
horizontal_average_v  = mean(v,     dims=(1, 2))
horizontal_average_b  = mean(b,     dims=(1, 2))
horizontal_average_e  = mean(e,     dims=(1, 2))
horizontal_average_vb = mean(vb,    dims=(1, 2))
horizontal_average_ζ² = mean(ζ²,    dims=(1, 2))
horizontal_average_b² = mean(b²,    dims=(1, 2))
horizontal_average_bz = mean(∂z(b), dims=(1, 2))

horizontal_averages = (
                       u  = horizontal_average_u,
                       v  = horizontal_average_v,
                       b  = horizontal_average_b,
                       e  = horizontal_average_e,
                       vb = horizontal_average_vb,
                       ζ² = horizontal_average_ζ²,
                       b² = horizontal_average_b²,
                       bz = horizontal_average_bz
                      )

volume_average_e  = mean(e,  dims=(1, 2, 3))
volume_average_vb = mean(vb, dims=(1, 2, 3))
volume_average_b² = mean(b², dims=(1, 2, 3))
volume_average_ζ² = mean(ζ², dims=(1, 2, 3))

volume_averages = (
                   e  = volume_average_e,
                   vb = volume_average_vb,
                   ζ² = volume_average_ζ²,
                   b² = volume_average_b²
                  )

save_interval_days = 4

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule=TimeInterval(floor(Int, simulation.stop_time/4)),
                 prefix=prefix * "_checkpointer")

simulation.output_writers[:xy_surface] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_xy_surface",
                     field_slicer = FieldSlicer(k=grid.Nz),
                            force = true)

k_subsurface = searchsortedfirst(znodes(Cell, grid), -100)

simulation.output_writers[:xy_subsurface] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_xy_subsurface",
                     field_slicer = FieldSlicer(k=k_subsurface),
                            force = true)

simulation.output_writers[:xy_middepth] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_xy_middepth",
                     field_slicer = FieldSlicer(k=round(Int, grid.Nz/2)),
                            force = true)

simulation.output_writers[:xy_bottom] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_xy_bottom",
                     field_slicer = FieldSlicer(k=1),
                            force = true)

simulation.output_writers[:xz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_xz",
                     field_slicer = FieldSlicer(j=1),
                            force = true)

simulation.output_writers[:yz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                         schedule = TimeInterval(save_interval_days * days),
                           prefix = prefix * "_yz",
                     field_slicer = FieldSlicer(i=1),
                            force = true)

simulation.output_writers[:horizontal_averages] =
    JLD2OutputWriter(model, horizontal_averages,
                     schedule = TimeInterval(save_interval_days * days),
                       prefix = prefix * "_profiles",
                        force = true)

simulation.output_writers[:volume_averages] =
    JLD2OutputWriter(model, volume_averages,
                     schedule = TimeInterval(save_interval_days * days),
                       prefix = prefix * "_volume_mean",
                        force = true)

# Press the big red button:

run!(simulation)

# # Visualizing Eady turbulence

pyplot() # pyplot backend is a bit nicer than GR

## Open the file with our data
file = jldopen(prefix * "_xy_surface.jld2")

f = file["coriolis/f"]

## Coordinate arrays
xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xδ, yδ, zδ = nodes((Cell, Cell, Cell), grid)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim, nlevels=30)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)
    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=nlevels), [cmax])
    end

    return levels
end

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    ## Load 3D fields from file
    t = file["timeseries/t/$iter"]

    R = file["timeseries/ζ/$iter"][:, :, 1] ./ f
    δ = file["timeseries/δ/$iter"][:, :, 1] ./ f
    u = file["timeseries/u/$iter"][:, :, 1]
    v = file["timeseries/v/$iter"][:, :, 1]
    w = file["timeseries/w/$iter"][:, :, 1]

    ũ = zeros(grid.Nx, grid.Ny)
    ṽ = zeros(grid.Nx, grid.Ny)

    @views ũ[1:end-1, :] = @. (u[1:end-1, :] + u[2:end, :]) / 2
    @views ũ[end, :]     = @. (u[1, :]       + u[end, :])   / 2

    @views ṽ[:, 1:end-1] = @. (v[:, 1:end-1] + v[:, 2:end]) / 2
    @views ṽ[:, end]     = @. (v[:, 1]       + v[:, end])   / 2

    s = @. sqrt(ũ^2 + ṽ^2)

    slim = 0.8 * maximum(abs, s) + 1e-9
    wlim = 0.5 * maximum(abs, w) + 1e-9
    Rlim = 0.8 * maximum(abs, R) + 1e-9

    wlevels = nice_divergent_levels(w, wlim)
    Rlevels = nice_divergent_levels(R, Rlim)

    smax = maximum(abs, s) + 1e-9

    slevels = vcat(range(0, stop=slim, length=30), [smax])

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f, max(δ / f) = %.3f \n",
                   i, iter, maximum(abs, R), maximum(abs, δ))

    R_plot = contourf(xζ, yζ, R';
                         colorbar = true,
                            color = :balance,
                      aspectratio = 1,
                           legend = false,
                            clims = (-Rlim, Rlim),
                           levels = Rlevels,
                            xlims = (0, grid.Lx),
                            ylims = (0, grid.Lx),
                           xlabel = "x (m)",
                           ylabel = "y (m)")

    w_plot = contourf(xδ, yδ, w';
                         colorbar = true,
                            color = :balance,
                      aspectratio = 1,
                           legend = false,
                            clims = (-wlim, wlim),
                           levels = wlevels,
                            xlims = (0, grid.Lx),
                            ylims = (0, grid.Lx),
                           xlabel = "x (m)",
                           ylabel = "y (m)")

    s_plot = contourf(xδ, yδ, s';
                         colorbar = true,
                            color = :thermal,
                      aspectratio = 1,
                           legend = false,
                            clims = (0, slim),
                           levels = slevels,
                            xlims = (0, grid.Lx),
                            ylims = (0, grid.Lx),
                           xlabel = "x (m)",
                           ylabel = "y (m)")

    ζ_title = @sprintf("ζ(z=0, t=%s) / f", prettytime(t))
    w_title = @sprintf("w(z=0, t=%s) (m s⁻¹)", prettytime(t))
    s_title = @sprintf("\$ \\sqrt{u^2 + v^2} \\, |_{z=0, t=%s} \$ (m s⁻¹)", prettytime(t))

    plot(R_plot, w_plot, s_plot, size = (1500, 600),
         layout = (1, 3),
          title = [ζ_title w_title s_title])

    iter == iterations[end] && close(file)
end

gif(anim, prefix * ".gif", fps = 8) # hide

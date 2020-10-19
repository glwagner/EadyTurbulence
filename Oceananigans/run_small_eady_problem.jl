using JLD2
using Plots
using Printf
using Statistics
using ArgParse
using Oceananigans
using Oceananigans.Grids
using Oceananigans.AbstractOperations
using Oceananigans.Fields: ComputedField, BackgroundField
using Oceananigans.Utils: minute, hour, day, GiB, prettytime
using Oceananigans.Advection: WENO5
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.OutputWriters: JLD2OutputWriter, FieldSlicer
using Oceananigans.Grids: x_domain, y_domain, z_domain # for nice domain limits

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
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

Nh = args["Nh"]
Nz = args["Nz"]
α_f = args["geostrophic-shear"]
stop_years = args["years"]
year = 365day

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 1e6), y=(0, 1e6), z=(-1e3, 0))

prefix = @sprintf("small_eady_problem_Nh%d_Nz%d_αf%.1e", grid.Nx, grid.Nz, α_f)

coriolis = FPlane(f=1e-4) # [s⁻¹]
                            
background_parameters = (       α = α_f * coriolis.f, # s⁻¹, geostrophic shear
                                f = coriolis.f,        # s⁻¹, Coriolis parameter
                           N_deep = sqrt(1e-5),        # s⁻¹, buoyancy frequency
                           N_surf = sqrt(1e-5),        # s⁻¹, buoyancy frequency
                          z_cline = -200,              # m, thermocline height
                          h_cline = 50,                # m, thermocline width
                               Lz = grid.Lz)           # m, ocean depth

@inline step(z, c, w) = (tanh((z-c) / w) + 1) / 2
@inline Bᴸ(z, N, Lz) = N^2 * (z + Lz)
@inline B_thermocline(z, N_deep, N_surf, z_cline, h_cline, Lz) =
    Bᴸ(z, N_deep, Lz) + (Bᴸ(z, N_surf, Lz) - Bᴸ(z, N_deep, Lz)) * step(z, z_cline, h_cline)

## Background fields are defined via functions of x, y, z, t, and optional parameters
U(x, y, z, t, p) = + p.α * (z + p.Lz)
#B(x, y, z, t, p) = - p.α * p.f * y + B_thermocline(z, p.N_deep, p.N_surf, p.z_cline, p.h_cline, p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y + p.N_deep^2 * z

U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

drag_coefficient = 1e-3

@inline drag_u(u, v, cᴰ) = - cᴰ * sqrt(u^2 + v^2) * u
@inline drag_v(u, v, cᴰ) = - cᴰ * sqrt(u^2 + v^2) * v

@inline bottom_drag_u(i, j, grid, clock, f, cᴰ) = @inbounds drag_u(f.u[i, j, 1], f.v[i, j, 1], cᴰ)
@inline bottom_drag_v(i, j, grid, clock, f, cᴰ) = @inbounds drag_v(f.u[i, j, 1], f.v[i, j, 1], cᴰ)
    
drag_bc_u = BoundaryCondition(Flux, bottom_drag_u, discrete_form=true, parameters=drag_coefficient)
drag_bc_v = BoundaryCondition(Flux, bottom_drag_v, discrete_form=true, parameters=drag_coefficient)

u_bcs = UVelocityBoundaryConditions(grid, bottom = drag_bc_u) 
v_bcs = VVelocityBoundaryConditions(grid, bottom = drag_bc_v)

κ₂z = 1e-4 # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e-2 / day * grid.Δx^4 # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

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
    boundary_conditions = (u=u_bcs, v=v_bcs)
)

## A noise function, damped at the top and bottom
Ξ(z) = randn() * z/grid.Lz * (z/grid.Lz + 1)

## Scales for the initial velocity and buoyancy
Ũ = 1e-1 * background_parameters.α * grid.Lz
B̃ = 1e-2 * background_parameters.α * coriolis.f

uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = B̃ * Ξ(z)

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

abs_max_Δt = grid.Nx >= 256 ? hour/3 : hour/2

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

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u)) # Vertical vorticity [s⁻¹]
δ = ComputedField(-∂z(w)) # Horizontal divergence, or ∂x(u) + ∂y(v) [s⁻¹]

## Eddy kinetic energy and buoyancy flux
f = coriolis.f

eddy_kinetic_energy = @at (Cell, Cell, Cell)  (u^2 + v^2 + w^2) / 2  
buoyancy_flux = @at (Cell, Cell, Cell)  v * b

#=
N = background_parameters.N

ω₁ = - ∂z(v)
ω₂ = ∂z(u)
ω₃ = ∂x(v) - ∂y(u)

available_pv = @at (Cell, Cell, Cell) begin
    ω₃ * (1 + ∂z(b) / N^2) + ∂z(f * b / N^2) + ω₁ * ∂x(b) / N^2 + ω₂ * ∂y(b) / N^2
end

 q = ComputedField(available_pv)
q² = ComputedField(available_pv^2)
=#

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

save_interval_days = 2

#=
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                      time_interval = year,
                                                             prefix = prefix * "_fields",
                                                              force = true)

simulation.output_writers[:xy_surface] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                      time_interval = save_interval_days * day,
                                                             prefix = prefix * "_xy_surface",
                                                       field_slicer = FieldSlicer(k=grid.Nz),
                                                              force = true)

simulation.output_writers[:xy_middepth] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                           time_interval = save_interval_days * day,
                                                                  prefix = prefix * "_xy_middepth",
                                                            field_slicer = FieldSlicer(k=round(Int, grid.Nz/2)),
                                                                   force = true)

simulation.output_writers[:xy_bottom] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                         time_interval = save_interval_days * day,
                                                                prefix = prefix * "_xy_bottom",
                                                          field_slicer = FieldSlicer(k=1),
                                                                 force = true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                  time_interval = save_interval_days * day,
                                                         prefix = prefix * "_xz",
                                                   field_slicer = FieldSlicer(j=1),
                                                          force = true)

simulation.output_writers[:yz] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                  time_interval = save_interval_days * day,
                                                               prefix = prefix * "_yz",
                                                         field_slicer = FieldSlicer(i=1),
                                                                force = true)

simulation.output_writers[:horizontal_averages] = JLD2OutputWriter(model, horizontal_averages,
                                                                   time_interval = save_interval_days * day,
                                                                          prefix = prefix * "_profiles",
                                                                           force = true)

simulation.output_writers[:volume_averages] = JLD2OutputWriter(model, volume_averages,
                                                               time_interval = save_interval_days * day,
                                                                      prefix = prefix * "_volume_mean",
                                                                       force = true)

# Press the big red button:

run!(simulation)
=#

# # Visualizing Eady turbulence

pyplot() # pyplot backend is a bit nicer than GR

## Open the file with our data
 surface_file = jldopen(prefix * "_xy_surface.jld2")
middepth_file = jldopen(prefix * "_xy_middepth.jld2")
  bottom_file = jldopen(prefix * "_xy_bottom.jld2")

f = surface_file["coriolis/f"]

## Coordinate arrays
xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xδ, yδ, zδ = nodes((Cell, Cell, Cell), grid)

## Extract a vector of iterations
iterations = parse.(Int, keys(surface_file["timeseries/t"]))

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
    t = surface_file["timeseries/t/$iter"]

    surface_R = surface_file["timeseries/ζ/$iter"][:, :, 1] ./ f
    surface_δ = surface_file["timeseries/δ/$iter"][:, :, 1] ./ f

    middepth_R = middepth_file["timeseries/ζ/$iter"][:, :, 1] ./ f
    middepth_δ = middepth_file["timeseries/δ/$iter"][:, :, 1] ./ f

    bottom_R = bottom_file["timeseries/ζ/$iter"][:, :, 1] ./ f
    bottom_δ = bottom_file["timeseries/δ/$iter"][:, :, 1] ./ f

    Rlim = 0.8 * maximum(abs, surface_R) + 1e-9
    δlim = 0.8 * maximum(abs, surface_δ) + 1e-9

    surface_Rlevels = nice_divergent_levels(surface_R, Rlim)
    surface_δlevels = nice_divergent_levels(surface_δ, δlim)
    middepth_Rlevels = nice_divergent_levels(middepth_R, Rlim)
    middepth_δlevels = nice_divergent_levels(middepth_δ, δlim)
    bottom_Rlevels = nice_divergent_levels(bottom_R, Rlim)
    bottom_δlevels = nice_divergent_levels(bottom_δ, δlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f, max(δ / f) = %.3f \n",
                   i, iter, maximum(abs, surface_R), maximum(abs, surface_δ))

    R_surface = contourf(xζ, yζ, surface_R';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = surface_Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    R_middepth = contourf(xζ, yζ, middepth_R';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = middepth_Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    R_bottom = contourf(xζ, yζ, bottom_R';
                       colorbar = true,
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = bottom_Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    δ_surface = contourf(xδ, yδ, surface_δ';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = surface_δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    δ_middepth = contourf(xδ, yδ, middepth_δ';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = middepth_δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    δ_bottom = contourf(xδ, yδ, bottom_δ';
                       colorbar = true,
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = bottom_δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    ζ0 = @sprintf("surface ζ/f(t=%s) (s⁻¹)", prettytime(t))
    δ0 = @sprintf("surface δ(t=%s) (s⁻¹)", prettytime(t))
    ζ1 = @sprintf("middepth ζ/f(t=%s) (s⁻¹)", prettytime(t))
    δ1 = @sprintf("middepth  δ(t=%s) (s⁻¹)", prettytime(t))
    ζ2 = @sprintf("bottom ζ/f(t=%s) (s⁻¹)", prettytime(t))
    δ2 = @sprintf("bottom δ(t=%s) (s⁻¹)", prettytime(t))

    plot(R_surface, R_middepth, R_bottom, δ_surface, δ_middepth, δ_bottom,
           size = (2000, 1000),
           link = :x,
         layout = (2, 3),
          title = [ζ0 ζ1 ζ2 δ0 δ1 δ2])

    iter == iterations[end] && (close(surface_file); close(middepth_file); close(bottom_file))
end

gif(anim, prefix * ".gif", fps = 8) # hide

using ArgParse
using Printf
using Statistics
using Plots
using JLD2
using Adapt

using Oceananigans
using Oceananigans.Grids
using Oceananigans.AbstractOperations
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction
using Oceananigans.Utils
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Fields

using Oceananigans.Operators: ℑxyᶜᶜᵃ, δyᵃᶠᵃ, δxᶠᵃᵃ
using Oceananigans.Diagnostics: AdvectiveCFL

Adapt.adapt_structure(to, dbf::DiscreteBoundaryFunction) =
    DiscreteBoundaryFunction(Adapt.adapt(to, dbf.func),
                             Adapt.adapt(to, dbf.parameters))

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

        "--stop-time"
            help = "Stop time for the simulation."
            default = 100.0
            arg_type = Float64

        "--bottom-bc"
            help = """The type of bottom boundary condition to use.
                      Options are:
                        * "linear-drag"
                        * "quadratic-drag"
                        * "pumping-velocity"
                   """
            default = "linear-drag"
            arg_type = String
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

Nx = args["Nh"]
Ny = args["Nh"]
Nz = args["Nz"]

#####
##### Parameters
#####

aspect_ratio = 1000
Lz = 1
Lx = aspect_ratio * Lz
Ly = aspect_ratio * Lz

  Ro = 0.25
  Ri = 1000 / Ro^2
  Pr = 1

   f = 1
   S = 0.25 * f
  N² = Ri * S^2

  D₁ = 0.0016
  D₂ = 1e-3
Ek_v = 1e-5
Ek_h = 3e-1
Ek_p = 1e-5

# Derived parameters

νv = Ek_v * f * Lz^2
νh = Ek_v * f * Lz^2
κv = νv / Pr
κh = νh / Pr
k₁ = D₁ * f * Lz
k₂ = D₂
kp = (Ek_p / 2)^2

#####
##### Background fields
#####
                            
U(x, y, z, t, p) = + p.S * (z + p.Lz)
B(x, y, z, t, p) = - p.S * p.f * y + p.N² * z

U_field = BackgroundField(U, parameters=(S=S, Lz=Lz))
B_field = BackgroundField(B, parameters=(S=S, N²=N², f=f, Lz=Lz))

#####
##### Build all the boundary conditions
#####

# Linear drag

@inline linear_drag_u(x, y, t, u, k₁) = - k₁ * u
@inline linear_drag_v(x, y, t, v, k₁) = - k₁ * v

linear_drag_u_bc = BoundaryCondition(Flux, linear_drag_u, field_dependencies=:u, parameters=k₁)
linear_drag_v_bc = BoundaryCondition(Flux, linear_drag_v, field_dependencies=:v, parameters=k₁)

# Quadratic drag

@inline quadratic_drag_u(x, y, t, u, v, k₂) = - k₂ * u * sqrt(u^2 + v^2)
@inline quadratic_drag_v(x, y, t, u, v, k₂) = - k₂ * v * sqrt(u^2 + v^2)

quadratic_drag_u_bc = BoundaryCondition(Flux, quadratic_drag_u, field_dependencies=(:u, :v), parameters=k₂)
quadratic_drag_v_bc = BoundaryCondition(Flux, quadratic_drag_v, field_dependencies=(:u, :v), parameters=k₂)
    
# Ekman pumping boundary condition on `w`

""" Returns the vertical component of vorticity. """
@inline function ζᶠᶠᶜ(i, j, k, grid, u, v)
    ∂x_v = δxᶠᵃᵃ(i, j, k, grid, v) / grid.Δx
    ∂y_u = δyᵃᶠᵃ(i, j, k, grid, u) / grid.Δy
    return ∂x_v - ∂y_u
end

@inline pumping_velocity(ζ, kp) = - ζ * kp

@inline function pumping_velocity(i, j, grid, clock, model_fields, kp)
    ζʷ = ℑxyᶜᶜᵃ(i, j, 1, grid, ζᶠᶠᶜ, model_fields.u, model_fields.v)
    return pumping_velocity(ζʷ, kp)
end

pumping_bc = BoundaryCondition(NormalFlow, pumping_velocity, discrete_form=true, parameters=kp)

##### 
##### Set boundary conditions: linear drag, quadratic drag, or an Ekman "pumping velocity"
##### 

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

bottom_bc = args["bottom-bc"]

if bottom_bc == "linear-drag"
    u_bcs = UVelocityBoundaryConditions(grid, bottom = linear_drag_u_bc)
    v_bcs = VVelocityBoundaryConditions(grid, bottom = linear_drag_v_bc)

    boundary_conditions = (u=u_bcs, v=v_bcs)

elseif bottom_bc == "quadratic-drag"
    u_bcs = UVelocityBoundaryConditions(grid, bottom = quadratic_drag_bc_u)
    v_bcs = VVelocityBoundaryConditions(grid, bottom = quadratic_drag_bc_v)

    boundary_conditions = (u=u_bcs, v=v_bcs)

elseif bottom_bc == "pumping-velocity"
    w_bcs = WVelocityBoundaryConditions(grid, bottom = pumping_bc)

    boundary_conditions = (w=w_bcs,)

else
    error("Bottom boundary condition $bottom_bc is not supported.")

end

Laplacian_diffusivity = AnisotropicDiffusivity(νh=νh, κh=κh, νz=νv, κz=κv)

#####
##### Model instantiation and initial condition
#####

prefix = @sprintf("non_dimensional_eady_%s_Nh%d_Nz%d", bottom_bc, grid.Nx, grid.Nz)

model = IncompressibleModel(
           architecture = GPU(),
                   grid = grid,
              advection = WENO5(),
            timestepper = :RungeKutta3,
               coriolis = FPlane(f=f),
                tracers = :b,
               buoyancy = BuoyancyTracer(),
      background_fields = (b=B_field, u=U_field),
                closure = Laplacian_diffusivity,
    boundary_conditions = boundary_conditions
)

# A noise function, damped at the top and bottom
Ξ(z) = randn() * z/grid.Lz * (z/grid.Lz + 1)

# Scales for the initial velocity and buoyancy
Ũ = S * Lz
B̃ = S * Lz * f

uᵢ(x, y, z) = 1e-1 * Ũ * Ξ(z)
vᵢ(x, y, z) = 1e-1 * Ũ * Ξ(z)
bᵢ(x, y, z) = 1e-2 * B̃ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

ū = sum(model.velocities.u.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
v̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)

model.velocities.u.data.parent .-= ū
model.velocities.v.data.parent .-= v̄

#####
##### Simulation construction
#####

max_Δt = min(0.3/f, grid.Δx / Ũ)

cfl = 0.5
bottom_bc == "quadratic-drag" && (cfl *= min(1, k₂ * grid.Δx / grid.Δz))

wizard = TimeStepWizard(cfl=cfl, Δt=0.1*max_Δt, max_change=1.1, max_Δt=max_Δt)

CFL = AdvectiveCFL(wizard)

start_time = time_ns()

mutable struct ProgressMessage{T}
    wall_time :: T
end

progress = ProgressMessage(time_ns())

function (p::ProgressMessage)(sim)

    @printf("i: % 6d, sim time: %.3e, wall time: % 10s, Δt: %.3e, CFL: %.2e\n",
            sim.model.clock.iteration,
            sim.model.clock.time,
            prettytime(1e-9 * (time_ns() - p.wall_time)),
            sim.Δt.Δt,
            CFL(sim.model))

    p.wall_time = time_ns()

    return nothing
end

stop_time = args["stop-time"]
simulation = Simulation(model,
                        Δt = wizard,
                        iteration_interval = 100,
                        stop_time = stop_time,
                        progress = progress)

#####
##### Output
#####

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u)) # Vertical vorticity [s⁻¹]
δ = ComputedField(-∂z(w)) # Horizontal divergence, or ∂x(u) + ∂y(v) [s⁻¹]

## Eddy kinetic energy and buoyancy flux
eddy_kinetic_energy = @at (Cell, Cell, Cell)  (u^2 + v^2 + w^2) / 2  
buoyancy_flux = @at (Cell, Cell, Cell) v * b

 e = ComputedField(eddy_kinetic_energy)
vb = ComputedField(buoyancy_flux)
b² = ComputedField(b^2)
ζ² = ComputedField(ζ^2)

profile_e  = mean(e,     dims=(1, 2))
profile_vb = mean(vb,    dims=(1, 2))
profile_ζ² = mean(ζ²,    dims=(1, 2))
profile_b² = mean(b²,    dims=(1, 2))
profile_bz = mean(∂z(b), dims=(1, 2))

volume_e  = mean(e,  dims=(1, 2, 3))
volume_vb = mean(vb, dims=(1, 2, 3))
volume_b² = mean(b², dims=(1, 2, 3))
volume_ζ² = mean(ζ², dims=(1, 2, 3))

pickup = false
fast_output_interval = floor(Int, stop_time/200)
force = pickup ? false : true

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule=TimeInterval(floor(Int, stop_time/10)), prefix = prefix * "_checkpointer")

simulation.output_writers[:xy_surface] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_xy_surface",
                     field_slicer = FieldSlicer(k=grid.Nz),
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:xy_bottom] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_xy_bottom",
                     field_slicer = FieldSlicer(k=1),
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:xy_near_bottom] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_xy_near_bottom",
                     field_slicer = FieldSlicer(k=2),
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:xz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_xz",
                     field_slicer = FieldSlicer(j=1),
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:yz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_yz",
                     field_slicer = FieldSlicer(i=1),
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, (e=profile_e, vb=profile_vb, ζ²=profile_ζ², b²=profile_b², bz=profile_bz),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_profiles",
                     max_filesize = 2GiB,
                     force = force)

simulation.output_writers[:volume] =
    JLD2OutputWriter(model, (e=volume_e, vb=volume_vb, ζ²=volume_ζ², b²=volume_b²),
                     schedule = TimeInterval(fast_output_interval),
                     prefix = prefix * "_volume_mean",
                     max_filesize = 2GiB,
                     force = force)

#####
##### Run the simulation
#####

run!(simulation, pickup=pickup)

#####
##### Visualizing Eady turbulence
#####

pyplot() # pyplot backend is a bit nicer than GR

profiles_file = jldopen(prefix * "_profiles.jld2")
volume_mean_file = jldopen(prefix * "_volume_mean.jld2")
surface_file = jldopen(prefix * "_xy_surface.jld2")
bottom_file = jldopen(prefix * "_xy_bottom.jld2")
near_bottom_file = jldopen(prefix * "_xy_near_bottom.jld2")

xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xw, yw, zw = nodes((Cell, Cell, Face), grid)

zc = znodes(Cell, grid)

iterations = parse.(Int, keys(surface_file["timeseries/t"]))

time = [surface_file["timeseries/t/$iter"] for iter in iterations]

 mean_e = [volume_mean_file["timeseries/e/$iter"][1, 1, 1]  for iter in iterations] 
mean_vb = [volume_mean_file["timeseries/vb/$iter"][1, 1, 1] for iter in iterations]
mean_ζ² = [volume_mean_file["timeseries/ζ²/$iter"][1, 1, 1] for iter in iterations]
mean_b² = [volume_mean_file["timeseries/b²/$iter"][1, 1, 1] for iter in iterations]

function divergent_levels(c, clim, nlevels=31)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return levels
end

normalize(ϕ) = ϕ

normalize_profile(ϕ) = ϕ ./ maximum(abs, ϕ)

@info "Making an animation from saved data..."

anim = @animate for (j, iter) in enumerate(iterations[10:10:end])

    i = j * 10

    ## Load 3D fields from file
    t = surface_file["timeseries/t/$iter"]

    surface_ζ = surface_file["timeseries/ζ/$iter"][:, :, 1]
    bottom_ζ = bottom_file["timeseries/ζ/$iter"][:, :, 1]
    bottom_w = near_bottom_file["timeseries/w/$iter"][:, :, 1]

    profile_e  = profiles_file["timeseries/e/$iter"][1, 1, :]
    profile_vb = profiles_file["timeseries/vb/$iter"][1, 1, :]
    profile_ζ² = profiles_file["timeseries/ζ²/$iter"][1, 1, :]
    profile_b² = profiles_file["timeseries/b²/$iter"][1, 1, :]

    ζlim = 0.6 * maximum(abs, surface_ζ) + 1e-9
    wlim = 0.8 * maximum(abs, bottom_w) + 1e-9

    ζlevels = divergent_levels(surface_ζ, ζlim)
    wlevels = divergent_levels(bottom_w, wlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(|ζ̃|) = %.3f\n",
                   i, iter, maximum(abs, surface_ζ))

    kwargs = (colorbar = true, color = :balance, aspectratio = 1, legend = false,
              xlims = (0, grid.Lx), ylims = (0, grid.Lx), xlabel = "x (m)", ylabel = "y (m)")
                           
    surface_ζ_plot = contourf(xζ, yζ, surface_ζ'; clims=(-ζlim, ζlim), levels=ζlevels, kwargs...)
    bottom_ζ_plot  = contourf(xζ, yζ, bottom_ζ';  clims=(-ζlim, ζlim), levels=ζlevels, kwargs...)
    bottom_w_plot  = contourf(xw, yw, bottom_w';  clims=(-wlim, wlim), levels=wlevels, kwargs...)

    volume_mean_plot = plot(time, normalize(mean_e);
                            linewidth=2, label="⟨e⟩", xlabel="time", ylabel="Volume mean")

    plot!(volume_mean_plot, time, normalize(mean_vb); linewidth=2, label="⟨vb⟩")
    plot!(volume_mean_plot, time, normalize(mean_ζ²); linewidth=2, label="⟨ζ²⟩")
    plot!(volume_mean_plot, time, normalize(mean_b²); linewidth=2, label="⟨b²⟩")
    plot!(volume_mean_plot, [1, 1] .* time[i], [0, 1]; linewidth=2, alpha=0.4, label=nothing)

    profiles_plot = plot(zc, normalize_profile(profile_e))
    plot!(profiles_plot, zc, normalize_profile(profile_vb))
    plot!(profiles_plot, zc, normalize_profile(profile_ζ²))
    plot!(profiles_plot, zc, normalize_profile(profile_b²))
              
    surface_ζ_title = @sprintf("ζ(z=0, t=%.3e)", t)
    bottom_ζ_title = @sprintf("ζ(z=-Lz, t=%.3e)", t)
    bottom_w_title = @sprintf("w(z=-Lz, t=%.3e) (m s⁻¹)", t)

    plot(surface_ζ_plot, profiles_plot, bottom_w_plot, volume_mean_plot,
           size = (1200, 1000),
         layout = (2, 2),
          title = [surface_ζ_title bottom_ζ_title bottom_w_title "Volume averages"])

    if iter == iterations[end]
        close(surface_file)
        close(bottom_file)
        close(near_bottom_file)
        close(profiles_file)
        close(volume_mean_file)
    end
end

gif(anim, prefix * ".gif", fps = 8) # hide

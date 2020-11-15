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
using Oceananigans.Fields: ComputedField, BackgroundField
using Oceananigans.Utils: minute, hour, day, GiB, prettytime
using Oceananigans.Advection: WENO5
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.OutputWriters
using Oceananigans.Grids: x_domain, y_domain, z_domain # for nice domain limits

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 64
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 16
            arg_type = Int

        "--geostrophic-shear"
            help = """The geostrophic shear non-dimensionalized by f."""
            default = 1e-4
            arg_type = Float64

        "--years"
            help = """The length of the simulation in years."""
            default = 0.1
            arg_type = Float64
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

# Parsed parameters
Nh = args["Nh"]
Nz = args["Nz"]
 α = args["geostrophic-shear"]

# Fixed parameters
 f = 1e-4 # s⁻¹
N² = 1e-5 # s⁻²
 L = 1e6  # m
 H = 1e3  # m

νh = κh = 1e4 # m² s⁻¹
νz = κz = 1e-2 # m² s⁻¹

λ = L / π

output_interval_days = 0.1

stop_years = args["years"]
year = 365day

# Output names
prefix = @sprintf("sinusoidal_eady_Nh%d_Nz%d_αf%.2e", Nh, Nz, α / f)
output_dir = joinpath("data", prefix)
mkpath(output_dir)

# Doin' stuff
grid = RegularCartesianGrid(size = (Nh, Nh, Nz), x = (0, L), y = (0, L), z = (-H, 0),
                            topology = (Periodic, Bounded, Bounded))

uᵢ(y, z) = + α * sin(y / λ) * (z + H)
bᵢ(y, z) = + α * f * λ * cos(y / λ) + N² * z

surface_Uz(x, y, t, p) = p.α * p.H * sin(y / p.λ)

b_bcs = TracerBoundaryConditions(grid)
                                 #top = GradientBoundaryCondition(N²),
                                 #bottom = GradientBoundaryCondition(N²))

u_bcs = UVelocityBoundaryConditions(grid,
                                    north = ValueBoundaryCondition(0),
                                    south = ValueBoundaryCondition(0),
                                    bottom = ValueBoundaryCondition(0))
                                    #top = GradientBoundaryCondition(surface_Uz, parameters=(α=α, H=H, λ=λ)),

# # Model instantiation

model = IncompressibleModel(
           architecture = GPU(),
                   grid = grid,
              advection = WENO5(),
            timestepper = :RungeKutta3,
               coriolis = FPlane(f=f),
                tracers = :b,
               buoyancy = BuoyancyTracer(),
                closure = AnisotropicDiffusivity(νh=νh, κh=νz, νz=κz, κz=κz),
    boundary_conditions = (u=u_bcs, b=b_bcs)
)


# A noise function, damped at the top and bottom
Ξ(z) = randn() * z/grid.Lz * (z/grid.Lz + 1)

uᵢ(x, y, z) = U(y, z)
bᵢ(x, y, z) = B(y, z) + α * f * L * 1e-3 * Ξ(z)

set!(model, u=uᵢ, b=bᵢ)

max_Δt = hour / 3

wizard = TimeStepWizard(cfl=1.0, Δt=max_Δt, max_change=1.1, max_Δt=max_Δt)

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

output_fields = merge(model.velocities, model.tracers, (ζ=ζ, δ=δ))

## Kinetic energy and buoyancy flux
e_op = @at (Cell, Cell, Cell) (u^2 + v^2 + w^2) / 2
e  = ComputedField(e_op)
b² = ComputedField(b^2)
ζ² = ComputedField(ζ^2)

volume_averages = (e = mean(e, dims=(1, 2, 3)),
                   ζ² = mean(ζ², dims=(1, 2, 3)),
                   b² = mean(b², dims=(1, 2, 3)))

simulation.output_writers[:volume_averages] =
    JLD2OutputWriter(model, volume_averages,
                     schedule = TimeInterval(day / 4),
                     dir = output_dir,
                     prefix = prefix * "_volume_averages",
                     force = true)

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        prefix = prefix * "_checkpointer",
                                                        dir = output_dir,
                                                        schedule = TimeInterval(year))

k_subsurface = searchsortedfirst(znodes(Cell, grid), -100)
k_middepth = searchsortedfirst(znodes(Cell, grid), -grid.Lz/2)

slicers = Dict(:xz => FieldSlicer(j=grid.Ny/2),
               :yz => FieldSlicer(i=1),
               :xy_surface => FieldSlicer(k=grid.Nz),
               :xy_subsurface => FieldSlicer(k=k_subsurface),
               :xy_middepth => FieldSlicer(k=k_middepth),
               :xy_nearbottom => FieldSlicer(k=2))

for (name, slicer) in slicers
    simulation.output_writers[name] =
        JLD2OutputWriter(model, output_fields,
                         schedule = TimeInterval(output_interval_days * day),
                         dir = output_dir,
                         prefix = prefix * string("_", name),
                         field_slicer = slicer,
                         force = true)
end

# Press the big red button:

run!(simulation)

# # Visualizing Eady turbulence

pyplot() # pyplot backend is a bit nicer than GR

## Open the file with our data
file = jldopen(joinpath(output_dir, prefix * "_xy_surface.jld2"))

f = file["coriolis/f"]

## Coordinate arrays
xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xw, yw, zw = nodes((Cell, Cell, Face), grid)

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim, nlevels=30)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return levels
end

# Now we're ready to animate.

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    local u
    local v
    local w

    ## Load 3D fields from file
    t = file["timeseries/t/$iter"]

    R = file["timeseries/ζ/$iter"][:, :, 1] ./ f
    u = file["timeseries/u/$iter"][:, :, 1]
    v = file["timeseries/v/$iter"][:, :, 1]
    w = file["timeseries/w/$iter"][:, :, 1]

    ũ = zeros(grid.Nx, grid.Ny)
    ṽ = zeros(grid.Nx, grid.Ny)

    @views ũ[1:end-1, :] = @. (u[1:end-1, :] + u[2:end, :]) / 2
    @views ũ[end, :]     = @. (u[1, :]       + u[end, :])   / 2

    @views @. ṽ = (v[:, 1:end-1] + v[:, 2:end]) / 2

    s = @. sqrt(ũ^2 + ṽ^2)

    slim = 0.2 #0.8 * maximum(abs, s) + 1e-9
    wlim = 1e-4 #0.8 * maximum(abs, w) + 1e-9
    Rlim = 0.4 #0.8 * maximum(abs, R) + 1e-9

    #wlevels = nice_divergent_levels(w, wlim)
    #Rlevels = nice_divergent_levels(R, Rlim)
    
    wlevels = range(-wlim, stop=wlim, length=30)
    Rlevels = range(-Rlim, stop=Rlim, length=30)

    smax = maximum(abs, s) + 1e-9
    slevels = collect(range(0, stop=slim, length=30))
    #slim < smax && push!(slevels, smax)

    @info @sprintf("Drawing frame %d from iteration %d: time: %s, max(ζ̃ / f) = %.3f \n",
                   i, iter, prettytime(t), maximum(abs, R))

    R_plot = contourf(xζ, yζ, clamp.(R, -Rlim, Rlim)';
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

    w_plot = contourf(xw, yw, clamp.(w, -wlim, wlim)';
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

    s_plot = contourf(xw, yw, clamp.(s, 0, slim)';
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

    ζ_title = @sprintf("ζ(z=0) at t=%s / f", prettytime(t))
    w_title = @sprintf("w(z=0) at t=%s (m s⁻¹)", prettytime(t))
    s_title = @sprintf("\$ \\sqrt{u^2 + v^2} \\, |_{z=0} at t=%s \$ (m s⁻¹)", prettytime(t))

    plot(R_plot, w_plot, s_plot,
           size = (1500, 600),
         layout = (1, 3),
          title = [ζ_title w_title s_title])

    iter == iterations[end] && close(file)
end

gif(anim, prefix * ".gif", fps = 8) # hide

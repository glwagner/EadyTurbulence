using JLD2
using Plots
using Printf
using Statistics
using ArgParse
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Utils: minute, hour, day, GiB, prettytime
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

        "--only-plot"
            help = """Just plot results, don't run anything."""
            default = false
            arg_type = Bool
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

# # Visualizing Eady turbulence

pyplot() # pyplot backend is a bit nicer than GR

## Open the file with our data
file_xy = jldopen(prefix * "_xy_surface.jld2")
file_xz = jldopen(prefix * "_xz.jld2")

f = file_xy["coriolis/f"]

## Coordinate arrays
xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xδ, yδ, zδ = nodes((Cell, Cell, Cell), grid)

## Extract a vector of iterations
iterations = parse.(Int, keys(file_xy["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim, nlevels=30)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)
    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=nlevels), [cmax])
    end

    return levels
end

function speed(u, v, grid)
    ũ = zeros(grid.Nx, grid.Ny)
    ṽ = zeros(grid.Nx, grid.Ny)

    @views ũ[1:end-1, :] = @. (u[1:end-1, :] + u[2:end, :]) / 2
    @views ũ[end, :]     = @. (u[1, :]       + u[end, :])   / 2

    @views ṽ[:, 1:end-1] = @. (v[:, 1:end-1] + v[:, 2:end]) / 2
    @views ṽ[:, end]     = @. (v[:, 1]       + v[:, end])   / 2

    s = @. sqrt(ũ^2 + ṽ^2)

    return s
end


# Now we're ready to animate.

@info "Making an animation from saved data..."

niters = length(iterations)
halfway = round(Int, niters/2)

anim = @animate for (i, iter) in enumerate(iterations[halfway:16:end])

    ## Load 3D fields from file
    t = file_xy["timeseries/t/$iter"]

    Rxy = file_xy["timeseries/ζ/$iter"][:, :, 1] ./ f
    wxy = file_xy["timeseries/w/$iter"][:, :, 1]

    Rxz = file_xz["timeseries/ζ/$iter"][:, 1, :] ./ f
    wxz = file_xz["timeseries/w/$iter"][:, 1, :]
    
    wlim = 1e-4
    Rlim = 0.4

    wlevels = nice_divergent_levels(wxz, wlim)
    Rlevels = nice_divergent_levels(Rxz, Rlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(|ζ| / f) = %.3f, max(|w|) = %.2e (m s⁻¹) \n",
                   i, iter, maximum(abs, Rxz), maximum(abs, wxz))

    xy_kwargs = (aspectratio = 1,
                      legend = false,
                       xlims = (0, grid.Lx),
                       ylims = (0, grid.Lx),
                      xlabel = "x (m)",
                      ylabel = "y (m)")

    xz_kwargs = (aspectratio = 10,
                    colorbar = true,
                      legend = false,
                       xlims = (0, grid.Lx),
                       ylims = (-grid.Lz, 0),
                      xlabel = "x (m)",
                      ylabel = "z (m)")

    Rxy_plot = contourf(xζ, yζ, Rxy';
                        color = :balance,
                        clims = (-Rlim, Rlim),
                       levels = Rlevels,
                       xy_kwargs...)

    wxy_plot = contourf(xδ, yδ, wxy';
                        color = :balance,
                        clims = (-wlim, wlim),
                       levels = wlevels,
                       xy_kwargs...)

    Rxz_plot = contourf(xζ, yζ, Rxz';
                        color = :balance,
                        clims = (-Rlim, Rlim),
                       levels = Rlevels,
                       xz_kwargs...)

    wxz_plot = contourf(xδ, yδ, wxz';
                        color = :balance,
                        clims = (-wlim, wlim),
                       levels = wlevels,
                       xz_kwargs...)

    Rxy_title = @sprintf("ζ(z=0, t=%s) / f", prettytime(t))
    wxy_title = @sprintf("w(z=0, t=%s) (m s⁻¹)", prettytime(t))

    Rxz_title = @sprintf("ζ(y=0, t=%s) / f", prettytime(t))
    wxz_title = @sprintf("w(y=0, t=%s) (m s⁻¹)", prettytime(t))

    plot(Rxy_plot, wxy_plot, Rxz_plot, wxz_plot,
           size = (1200, 1000),
         layout = (2, 2),
          title = [Rxy_title wxy_title Rxz_title wxz_title])

    iter == iterations[end] && close(file)
end

gif(anim, prefix * "_two_depths.gif", fps = 8) # hide

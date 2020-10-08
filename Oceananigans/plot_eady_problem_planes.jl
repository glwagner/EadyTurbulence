# # Visualizing Eady turbulence
#
# We animate the results by opening the JLD2 file, extracting data for
# the iterations we ended up saving at, and ploting slices of the saved
# fields. We prepare for animating the flow by creating coordinate arrays,
# opening the file, building a vector of the iterations that we saved
# data at, and defining a function for computing colorbar limits: 

using JLD2, Plots, Printf, Oceananigans, Oceananigans.Grids

using Oceananigans.Grids: x_domain, y_domain, z_domain # for nice domain limits

pyplot() # pyplot backend is a bit nicer than GR

Nx = 128
Nz = 128

#prefix = @sprintf("eady_turbulence_Nh%d_Nz%d", Nx, Nz)
prefix = @sprintf("small_eady_turbulence_Nh%d_Nz%d", Nx, Nz)

## Open the file with our data
 surface_file = jldopen(prefix * "_xy_surface.jld2")
middepth_file = jldopen(prefix * "_xy_middepth.jld2")
  bottom_file = jldopen(prefix * "_xy_bottom.jld2")

Nx = surface_file["grid/Nx"]
Ny = surface_file["grid/Ny"]
Nz = surface_file["grid/Nz"]

Lx = surface_file["grid/Lx"]
Ly = surface_file["grid/Ly"]
Lz = surface_file["grid/Lz"]

f = surface_file["coriolis/f"]

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

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

anim = @animate for (i, iter) in enumerate(iterations[1:553])

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

    Rlevels = nice_divergent_levels(surface_R, Rlim)
    δlevels = nice_divergent_levels(surface_δ, δlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f, max(δ / f) = %.3f \n",
                   i, iter, maximum(abs, surface_R), maximum(abs, surface_δ))

    R_surface = contourf(xζ, yζ, surface_R';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    R_middepth = contourf(xζ, yζ, middepth_R';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = Rlevels,
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
                         levels = Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    δ_surface = contourf(xδ, yδ, surface_δ';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    δ_middepth = contourf(xδ, yδ, middepth_δ';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = δlevels,
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
                         levels = δlevels,
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

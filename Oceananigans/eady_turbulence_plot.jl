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

## Open the file with our data
xy_file = jldopen("eady_turbulence_surface.jld2")
xz_file = jldopen("eady_turbulence_xzslices.jld2")

Nx = xy_file["grid/Nx"]
Ny = xy_file["grid/Ny"]
Nz = xy_file["grid/Nz"]

Lx = xy_file["grid/Lx"]
Ly = xy_file["grid/Ly"]
Lz = xy_file["grid/Lz"]

f = xy_file["coriolis/f"]

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

## Coordinate arrays
xζ, yζ, zζ = nodes((Face, Face, Cell), grid)
xδ, yδ, zδ = nodes((Cell, Cell, Cell), grid)

## Extract a vector of iterations
iterations = parse.(Int, keys(xy_file["timeseries/t"]))

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
    t = xy_file["timeseries/t/$iter"]
    surface_R = xy_file["timeseries/ζ/$iter"][:, :, 1] ./ f
    surface_δ = xy_file["timeseries/δ/$iter"][:, :, 1]

    slice_R = xz_file["timeseries/ζ/$iter"][:, 1, :] ./ f
    slice_δ = xz_file["timeseries/δ/$iter"][:, 1, :]

    @show maximum(slice_R)

    Rlim = 0.5 * maximum(abs, surface_R) + 1e-9
    δlim = 0.5 * maximum(abs, surface_δ) + 1e-9

    Rlevels = nice_divergent_levels(surface_R, Rlim)
    δlevels = nice_divergent_levels(surface_δ, δlim)

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f \n",
                   i, iter, maximum(abs, surface_R))

    R_xy = contourf(xζ, yζ, surface_R';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")
    
    δ_xy = contourf(xδ, yδ, surface_δ';
                          color = :balance,
                    aspectratio = 1,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (0, grid.Lx),
                         xlabel = "x (m)",
                         ylabel = "y (m)")

    R_xz = contourf(xζ, zζ, slice_R';
                    aspectratio = grid.Lx / grid.Lz * 0.5,
                          color = :balance,
                         legend = false,
                          clims = (-Rlim, Rlim),
                         levels = Rlevels,
                          xlims = (0, grid.Lx),
                          ylims = (-grid.Lz, 0),
                         xlabel = "x (m)",
                         ylabel = "z (m)")

    δ_xz = contourf(xδ, zδ, slice_δ';
                    aspectratio = grid.Lx / grid.Lz * 0.5,
                          color = :balance,
                         legend = false,
                          clims = (-δlim, δlim),
                         levels = δlevels,
                          xlims = (0, grid.Lx),
                          ylims = (-grid.Lz, 0),
                         xlabel = "x (m)",
                         ylabel = "z (m)")

    plot(R_xy, δ_xy, R_xz, δ_xz,
           size = (1000, 800),
           link = :x,
         layout = Plots.grid(2, 2, heights=[0.5, 0.5, 0.2, 0.2]),
          title = [@sprintf("ζ/f(t=%s) (s⁻¹)", prettytime(t)) @sprintf("δ(t=%s) (s⁻¹)", prettytime(t)) "" ""])

    iter == iterations[end] && (close(xy_file); close(xz_file))
end

gif(anim, "eady_turbulence.gif", fps = 8) # hide

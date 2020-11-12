using JLD2
using Plots
using Oceananigans
using Oceananigans.Grids

filename = "continued_eady_problem_u_NIW1.0_Nh384_Nz96_profiles.jld2"

file = jldopen(filename)

Nx = file["grid/Nx"]
Ny = file["grid/Ny"]
Nz = file["grid/Nz"]

Lx = file["grid/Lx"]
Ly = file["grid/Ly"]
Lz = file["grid/Lz"]

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

z = znodes(Cell, grid)

iterations = parse.(Int, keys(file["timeseries/t"]))

speed = zeros(length(iterations), Nz)

for (i, iter) in enumerate(iterations)
    U = file["timeseries/u/$iter"][1, 1, :]
    V = file["timeseries/v/$iter"][1, 1, :]

    @. speed[i, :] = sqrt(U^2 + V^2)
end

time = [file["timeseries/t/$iter"] for iter in iterations]

m = contourf(time .* 1e-4 / 2π, z, speed',
             clims=(0, 0.5),
             levels = vcat(range(0, stop=0.5, length=21), [maximum(abs, speed)]),
             linewidth=0.1,
             linealpha=0.4,
             xlabel="Inertial periods", ylabel="z (m)",
             title="\$ \\sqrt{U^2 + V^2} \\, \\mathrm{m \\, s^{-1}} \$")
           
display(m)

#=
anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter..."

    U = file["timeseries/u/$iter"][1, 1, :]
    V = file["timeseries/v/$iter"][1, 1, :]

    s = @. sqrt(U^2 + V^2)

    p1 = plot(U, z, label="U", xlims=(-1, 1), xlabel="Velocity (m s⁻¹)", ylabel="z (m)")
    plot!(p1, V, z, label="V")

    p2 = plot(s, z, label=nothing, xlims=(0, 1.2), xlabel="Speed (m s⁻¹)", ylabel="z (m)")
    
    plot(p1, p2, layout=(1, 2))
end

gif(anim, "profiles.gif", fps=8)
=#

close(file)

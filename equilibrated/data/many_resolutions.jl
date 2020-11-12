using JLD2
using Plots

filenames =  [

    "non_dimensional_eady_linear-drag_vEkman1.0e-05_Nh192_Nz32_volume_mean.jld2",
    "non_dimensional_eady_linear-drag_vEkman1.0e-05_Nh256_Nz32_volume_mean.jld2",
    "non_dimensional_eady_pumping-velocity_vEkman1.0e-05_Nh192_Nz32_volume_mean.jld2",
    "non_dimensional_eady_pumping-velocity_vEkman1.0e-05_Nh256_Nz32_volume_mean.jld2",

    ]

function resolution(filepath)
    file = jldopen(filepath)
    Nx = file["grid/Nx"]
    Nz = file["grid/Nz"]
    return Nx, Nz
end

function simulation_label(filepath)

    prefix = "non_dimensional_eady_"
    suffix = "_vEkman1.0e-05_Nh256_Nz32_volume_mean.jld2"
    name = filepath[length(prefix)+1:end-length(suffix)]

    name = replace(name, "-" => " ")

    return name
end


function get_time_series(filepath, var)
    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))

    time = [file["timeseries/t/$iter"] for iter in iterations]
    timeseries = [file["timeseries/$var/$iter"][1, 1, 1] for iter in iterations]

    close(file)

    return time, timeseries
end

time, eke = get_time_series(filenames[1], "e")
time, vb = get_time_series(filenames[1], "vb")
time, ζ² = get_time_series(filenames[1], "ζ²")
Nx, Nz = resolution(filenames[1])
name = simulation_label(filenames[1])

vb_plot = plot(time, vb, label="$name, Nʰ=$Nx, Nᶻ=$Nz", xlabel="Time", ylabel="Volume-averaged buoyancy flux",
               ylims = (-10, 20))

eke_plot = plot(time, eke, label="$name, Nʰ=$Nx, Nᶻ=$Nz", xlabel="Time", ylabel="Volume-averaged buoyancy flux",
                ylims = (-10, 20))

ζ²_plot = plot(time, ζ², label="$name, Nʰ=$Nx, Nᶻ=$Nz", xlabel="Time", ylabel="Volume-averaged buoyancy flux",
               ylims = (-10, 20))

for filename in filenames[2:end]

    local time
    local name
    local eke
    local ζ²
    local vb

    time, eke = get_time_series(filename, "e")
    time, vb = get_time_series(filename, "vb")
    time, ζ² = get_time_series(filename, "ζ²")
    Nx, Nz = resolution(filename)
    name = simulation_label(filename)
    
    plot!(vb_plot, time, vb, label="$name, Nʰ=$Nx, Nᶻ=$Nz")
    plot!(eke_plot, time, eke, label="$name, Nʰ=$Nx, Nᶻ=$Nz")
    plot!(ζ²_plot, time, ζ², label="$name, Nʰ=$Nx, Nᶻ=$Nz")
end

combined_plot = plot(vb_plot, eke_plot, ζ²_plot, layout=(3, 1))

display(combined_plot)

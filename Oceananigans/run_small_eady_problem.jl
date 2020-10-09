using Printf
using Statistics
using Oceananigans
using Oceananigans.Grids
using Oceananigans.AbstractOperations
using Oceananigans.Fields: ComputedField, BackgroundField
using Oceananigans.Utils: minute, hour, day, GiB
using Oceananigans.Advection: WENO5
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.OutputWriters: JLD2OutputWriter, FieldSlicer

grid = RegularCartesianGrid(size=(256, 256, 128), x=(0, 1e6), y=(0, 1e6), z=(-1e3, 0))

prefix = @sprintf("small_eady_problem_Nh%d_Nz%d", grid.Nx, grid.Nz)

coriolis = FPlane(f=1e-4) # [s⁻¹]
                            
background_parameters = (       α = 0.25 * coriolis.f, # s⁻¹, geostrophic shear
                                f = coriolis.f,        # s⁻¹, Coriolis parameter
                           N_deep = sqrt(1e-6),        # s⁻¹, buoyancy frequency
                           N_surf = sqrt(1e-5),        # s⁻¹, buoyancy frequency
                          z_cline = -200,        # s⁻¹, buoyancy frequency
                          h_cline = 50,        # s⁻¹, buoyancy frequency
                               Lz = grid.Lz)           # m, ocean depth

@inline step(z, c, w) = (tanh((z-c) / w) + 1) / 2
@inline Bᴸ(z, N, Lz) = N^2 * (z + Lz)
@inline B_thermocline(z, N_deep, N_surf, z_cline, h_cline, Lz) =
    Bᴸ(z, N_deep, Lz) + (Bᴸ(z, N_surf, Lz) - Bᴸ(z, N_deep, Lz)) * step(z, z_cline, h_cline)

## Background fields are defined via functions of x, y, z, t, and optional parameters
U(x, y, z, t, p) = + p.α * (z + p.Lz)
B(x, y, z, t, p) = - p.α * p.f * y + B_thermocline(z, p.N_deep, p.N_surf, p.z_cline, p.h_cline, p.Lz)
#B(x, y, z, t, p) = - p.α * p.f * y + p.N_deep^2 * z

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

max_Δt = min(hour / 2, grid.Δx / Ū, 0.5 * grid.Δx^4 / κ₄h, 0.5 * grid.Δz^2 / κ₂z)

cfl = 1.0
cfl = cfl * min(cfl, drag_coefficient * grid.Δx / grid.Δz)

wizard = TimeStepWizard(cfl=cfl, Δt=0.1*max_Δt, max_change=1.1, max_Δt=max_Δt)

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
                                                     stop_time = 10 * 365day,
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

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                      time_interval = 30day,
                                                             prefix = prefix * "_fields",
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:xy_surface] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                      time_interval = 84hour,
                                                             prefix = prefix * "_xy_surface",
                                                       field_slicer = FieldSlicer(k=grid.Nz),
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:xy_middepth] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                        time_interval = 84hour,
                                                               prefix = prefix * "_xy_middepth",
                                                         field_slicer = FieldSlicer(k=round(Int, grid.Nz/2)),
                                                         max_filesize = 2GiB,
                                                                force = true)

simulation.output_writers[:xy_bottom] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                      time_interval = 84hour,
                                                             prefix = prefix * "_xy_bottom",
                                                       field_slicer = FieldSlicer(k=1),
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                        time_interval = 84hour,
                                                               prefix = prefix * "_xz",
                                                         field_slicer = FieldSlicer(j=1),
                                                         max_filesize = 2GiB,
                                                                force = true)

simulation.output_writers[:yz] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (ζ=ζ, δ=δ)),
                                                        time_interval = 84hour,
                                                               prefix = prefix * "_yz",
                                                         field_slicer = FieldSlicer(i=1),
                                                         max_filesize = 2GiB,
                                                                force = true)

simulation.output_writers[:horizontal_averages] = JLD2OutputWriter(model, horizontal_averages,
                                                                   time_interval = 84hour,
                                                                          prefix = prefix * "_profiles",
                                                                    max_filesize = 2GiB,
                                                                           force = true)

simulation.output_writers[:volume_averages] = JLD2OutputWriter(model, volume_averages,
                                                        time_interval = 84hour,
                                                               prefix = prefix * "_volume_mean",
                                                         max_filesize = 2GiB,
                                                                force = true)

# Press the big red button:

run!(simulation)

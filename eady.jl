using Oceananigans, PyPlot, Random, Printf

using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: ∂x_faa, ∂x_caa, ▶x_faa, ▶y_aca, ▶x_caa, ▶xz_fac

# Resolution
Nh = 128
Nz = 8 

# Domain size
Lh = 1000e3
Lz = 1000

# Diffusivity
Δh = Lh / Nh
Δz = Lz / Nz

@show κh = Δh^2 / 0.25day
κv = (Δz / Δh)^2 * κh

# Physical parameters
 f = 1e-4     # Coriolis parameter
N² = 1e-6     # Stratification in the "halocline"
 α = 1e-2     # [s⁻¹] background shear

# Simulation end time
end_time = 30day

#####
##### Boundary conditions
#####

# Parameters
 μ = 1e-6     # [s⁻¹] drag coefficient
Cτ = 0.4 # "Von Karman constant"
z₀ = 1.0 # Must be smaller than half the grid spacing... ?

@inline function τ₁₃_MoninObukhov(i, j, grid, time, iter, U, C, p)
    s = sqrt(U.u[1]^2 + U.v[1]^2)
    return - p.Cτ * s * U.u[1] / log(grid.Δz / (2*p.z₀))^2
end

@inline function τ₂₃_MoninObukhov(i, j, grid, time, iter, U, C, p)
    s = sqrt(U.u[1]^2 + U.v[1]^2)
    return - p.Cτ * s * U.v[1] / log(grid.Δz / (2*p.z₀))^2
end

@inline τ₁₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * grid.Lz * U.u[i, j, 1]
@inline τ₂₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * grid.Lz * U.v[i, j, 1]

u_bcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_linear_drag))
v_bcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_linear_drag))

#####
##### "Forcing" terms associated with background balanced flow
#####

#=
# Can code up more general functions, assuming ψ is defined at (a, c, c).

  ∂y_ψ(i, j, k, grid, p) = @inbounds -p.α * grid.zC[k]
∂y∂z_ψ(i, j, k, grid, p) = -p.α
 ∂²z_ψ(i, j, k, grid, p) = 0
=#

#
# Total momentum forcing is
#
# Fu = -(α w + α z ∂ₓu) x̂ - α z ∂ₓv ŷ - α z ∂ₓ w ẑ
#
# while buoyancy forcing is
#
# Fb = - α z ∂ₓb + α f v
#
# Forcing must respect the staggered grid.

# Fu = -α w - α z ∂ₓu is applied at location (f, c, c).  
Fu(i, j, k, grid, time, U, C, p) = @inbounds (
    - p.α * ▶xz_fac(i, j, k, grid, U.w)
    - p.α * grid.zC[k] * ∂x_faa(i, j, k, grid, ▶x_caa, U.u))

# Fv = - α z ∂ₓv is applied at location (c, f, c).  
Fv(i, j, k, grid, time, U, C, p) = @inbounds -p.α * grid.zC[k] * ∂x_caa(i, j, k, grid, ▶x_faa, U.v)

# Fw = - α z ∂ₓw is applied at location (c, c, f).  
Fw(i, j, k, grid, time, U, C, p) = @inbounds -p.α * grid.zF[k] * ∂x_caa(i, j, k, grid, ▶x_faa, U.w)

# Fb = - α z ∂ₓb + α f v
Fb(i, j, k, grid, time, U, C, p) = @inbounds (
    - p.α * grid.zC[k] * ∂x_caa(i, j, k, grid, ▶x_faa, C.b)
    + p.f * p.α * ▶y_aca(i, j, k, grid, U.v))

#####
##### Model instantiation
#####

model = Model(
         architecture = CPU(),
                 grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(-Lh/2, Lh/2), y=(-Lh/2, Lh/2), z=(-Lz, 0)),
             coriolis = FPlane(f=f),
             buoyancy = BuoyancyTracer(),
              tracers = :b,
              closure = ConstantAnisotropicDiffusivity(νh=κh, κh=κh, νv=κv, κv=κv),
              forcing = ModelForcing(u=Fu, v=Fv, w=Fw, b=Fb),
  #boundary_conditions = BoundaryConditions(u=u_bcs, v=v_bcs),
           parameters = (α=α, f=f, μ=μ, Cτ=Cτ, z₀=z₀)
)

# Initial condition
Ξ(z) = rand() * z * (z + model.grid.Lz)

b₀(x, y, z) = N² * z + 1e-6 * Ξ(z) * α * f * model.grid.Ly
u₀(x, y, z) = 1e-6 * Ξ(z) * α * model.grid.Lz

set!(model, u=u₀, v=u₀, b=b₀)

# A diagnostic that returns the maximum absolute value of `w` by calling
# `wmax(model)`:

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

# Set up output
fields_to_output = merge(model.velocities, (b=model.tracers.b,))
output_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=day, prefix="beaufort_gyre",
                                 force=true)

# Create a figure
close("all")
fig, axs = subplots(ncols=3, figsize=(16, 6))

function makeplot!(fig, axs, model)

    fig.suptitle("\$ t = \$ $(prettytime(model.clock.time))")

    sca(axs[1]); cla()
    title("\$ u(x, y, z=0) \$")
    imshow(interior(model.velocities.u)[:, :, Nz])

    sca(axs[2]); cla()
    title("\$ v(x, y, z=0) \$")
    imshow(interior(model.velocities.v)[:, :, Nz])

    kplot = Nz - 2
    sca(axs[3]); cla()
    title("\$ w(x, y, z=$(model.grid.zF[kplot]))")
    imshow(interior(model.velocities.w)[:, :, kplot])
    
    [ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false) for ax in axs]

    return nothing
end

wizard = TimeStepWizard(cfl=0.05, Δt=20.0, max_change=1.1, max_Δt=10minute)

while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = @elapsed time_step!(model, 10, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            umax(model), vmax(model), wmax(model), prettytime(walltime))

    if model.clock.iteration % 100 == 0
        model.architecture == CPU() && makeplot!(fig, axs, model)
    end
end

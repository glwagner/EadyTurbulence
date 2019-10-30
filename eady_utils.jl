using Oceananigans

using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: ∂x_faa, ∂x_caa, ▶x_faa, ▶y_aca, ▶x_caa, ▶xz_fac

function print_eady_parameters(N², f, α, H)
    deformation_radius = sqrt(N²) * H / f
    baroclinic_growth_rate = sqrt(N²) / (α * f)

    @printf("Deformation radius: %.2f km \nBaroclinic growth rate: %.2f days\n", 
            deformation_radius * 1e-3, baroclinic_growth_rate / day)

    return nothing
end

#
# Total momentum forcing is
#
# Fu = -(α w + α (z + H) ∂ₓu) x̂ - α z ∂ₓv ŷ - α z ∂ₓ w ẑ
#
# while buoyancy forcing is
#
# Fb = - α (z + H) ∂ₓb + α f v
#
# Forcing must respect the staggered grid.

# Fu = -α w - α (z + H) ∂ₓu is applied at location (f, c, c).  
Fu_eady(i, j, k, grid, time, U, C, p) = @inbounds (
    - p.α * ▶xz_fac(i, j, k, grid, U.w)
    - p.α * (grid.zC[k] + grid.Lz) * ∂x_faa(i, j, k, grid, ▶x_caa, U.u))

# Fv = - α (z + H) ∂ₓv is applied at location (c, f, c).  
Fv_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zC[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, U.v)

# Fw = - α (z + H) ∂ₓw is applied at location (c, c, f).  
Fw_eady(i, j, k, grid, time, U, C, p) = @inbounds -p.α * (grid.zF[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, U.w)

# Fb = - α z ∂ₓb + α f v
Fb_eady(i, j, k, grid, time, U, C, p) = @inbounds (- p.α * (grid.zC[k] + grid.Lz) * ∂x_caa(i, j, k, grid, ▶x_faa, C.b)
                                                   + p.f * p.α * ▶y_aca(i, j, k, grid, U.v))

#####
##### Boundary conditions
#####
#
@inline τ₁₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.u[i, j, 1]
@inline τ₂₃_linear_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.μ * p.H * U.v[i, j, 1]

@inline τ₁₃_quadratic_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.Cd * U.u[i, j, 1]^2
@inline τ₂₃_quadratic_drag(i, j, grid, time, iter, U, C, p) = @inbounds p.Cd * U.v[i, j, 1]^2

@inline function τ₁₃_Monin_Obukhov(i, j, grid, time, iter, U, C, p)
    @inbounds s = sqrt(U.u[i, j, 1]^2 + U.v[i, j, 1]^2)
    return @inbounds - p.Cτ * s * U.u[i, j, 1] / log(grid.Δz / (2*p.z₀))^2
end

@inline function τ₂₃_Monin_Obukhov(i, j, grid, time, iter, U, C, p)
    @inbounds s = sqrt(U.u[i, j, 1]^2 + U.v[i, j, 1]^2)
    return @inbounds - p.Cτ * s * U.v[i, j, 1] / log(grid.Δz / (2*p.z₀))^2
end

function linear_drag_boundary_conditions(; N², μ, H)
    ubcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_linear_drag))
    vbcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_linear_drag))
    bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, N²), 
                                   bottom = BoundaryCondition(Gradient, N²))
    parameters = (μ=μ, H=H)
    return ubcs, vbcs, bbcs, parameters
end

function quadratic_drag_boundary_conditions(; N², Cd)
    ubcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_quadratic_drag))
    vbcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_quadratic_drag))
    bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, N²), 
                                   bottom = BoundaryCondition(Gradient, N²))
    parameters = (Cd=Cd,)
    return ubcs, vbcs, bbcs, parameters
end

function Monin_Obukhov_boundary_conditions(; N², Von_Karman_constant=0.41, roughness_length=10.0)
    ubcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₁₃_Monin_Obukhov))
    vbcs = HorizontallyPeriodicBCs(bottom = BoundaryCondition(Flux, τ₂₃_Monin_Obukhov))
    bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, N²), 
                                   bottom = BoundaryCondition(Gradient, N²))
    parameters = (z₀=roughness_length, Cτ=Von_Karman_constant)
    return ubcs, vbcs, bbcs, parameters
end



function background_geostrophic_flow_forcing(; geostrophic_shear, f)
    forcing = ModelForcing(u=Fu_eady, v=Fv_eady, w=Fw_eady, b=Fb_eady)
    forcing_parameters = (α=α, f=f)
    return forcing, forcing_parameters
end

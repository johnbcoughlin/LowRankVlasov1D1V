module LowRankVlasov1D1V

using Parameters
using LinearAlgebra
using LinearMaps: LinearMap
using KronOperators

include("alloc.jl")
include("utils.jl")
include("hyperbolic.jl")
include("elliptic.jl")
include("initial_conditions.jl")

export step!, expand_f

struct Domain
    nx::Int
    dx::Float64
    x_min::Float64
    x_max::Float64
    x::Vector{Float64}
    Dx::Derivatives
    Ax::Averages

    nv::Int
    dv::Float64
    v_min::Float64
    v_max::Float64
    v::Vector{Float64}
    Dv::Derivatives
    Av::Averages
end

include("fulltensor.jl")

mutable struct Simulation
    domain::Domain

    X::Matrix{Float64}
    S::Matrix{Float64}
    V::Matrix{Float64}

    E::Vector{Float64}

    η::Vector{Float64}

    q::Float64

    r::Int

    arena::Arena
end

include("l_then_k.jl")
include("unconventional.jl")

expand_f(sim) = sim.X * sim.S * sim.V'

function step!(sim::Simulation, Δt; debug=false)
    @unpack arena, domain, X, S, V, E, η, q, r = sim
    @unpack nx, dx, dv, v, Dx, Dv, Ax, Av = domain

    ρ = q * X * S * sum(V, dims=1)' .* dv
    E = solve_poisson(Dx, ρ - η)

    # K step
    K = alloc_f64!(arena, :K, size(X))
    mul!(K, X, S)
    A¹ = r_r_matrix_V(v .* V, V, domain, r)
    A² = r_r_matrix_V(V, Dv.centered * V, domain, r)

    if debug
        display(eigvals(A¹))
    end

    dK = -Δt * flux_limited_linear_hyperbolic(A¹, K, Dx, Ax, true, Δt, dx, arena, :K)
    KE = alloc_f64!(arena, :KE, size(K))
    @. KE = K * E * q
    dK -= Δt * (KE) * A²'

    K = K + dK
    X, S = qr(K)
    X = Matrix(X) / sqrt(dx)
    S = S * sqrt(dx)

    f_after_K = X * S * V'
    after_K = alloc_f64!(arena, :after_K, size(f_after_K))
    after_K .= f_after_K

    C¹ = r_r_matrix_X(X, Dx.centered * X, domain, r)
    C² = r_r_matrix_X(X, q * E .* X, domain, r)

    B¹ = LinearMap(A¹) ⊗ LinearMap(C¹)
    B² = LinearMap(A²) ⊗ LinearMap(C²')

    #B = alloc_f64!(arena, :B, (r^2, r^2))
    #@. B = Δt * (B¹ + B²)
    B = Δt * (B¹ + B²)

    # S step
    S = S + reshape(B * vec(S), (r, r))

    f_after_S = X * S * V'
    after_S = alloc_f64!(arena, :after_S, size(f_after_S))
    after_S .= f_after_S

    # L step
    L = (S * V')'

    dL = -Δt * flux_limited_linear_hyperbolic(C², L, Dv, Av, true, Δt, dv, arena, :L)
    dL -= Δt * (L .* v) * C¹'

    L = L + dL

    V, St = qr(L)
    V = Matrix(V) / sqrt(dv)
    S = (St * sqrt(dv))'

    f_after_L = X * S * V'
    after_L = alloc_f64!(arena, :after_L, size(f_after_L))
    after_L .= f_after_L

    sim.X = X
    sim.S = S
    sim.V = V
    sim.E = E
end

function K_ode(K, p, t)
    @unpack A¹, Dx, Ax, Δt, dx, arena = p
    -flux_limited_linear_hyperbolic(A¹, K, Dx, Ax, true, Δt, dx, arena, :K)
end

function L_ode(L, p, t)
    @unpack C², Dv, Av, Δt, dv, arena = p
    -flux_limited_linear_hyperbolic(C², L, Dv, Av, true, Δt, dv, arena, :L)
end

function r_r_matrix_X(X1, X2, domain, r)
    result = zeros(r, r)
    for i in 1:r
        for k in 1:r
            tik = 0.0
            for x in axes(X1, 1)
                tik += X1[x, i] * X2[x, k]
            end
            result[i,k] = tik * domain.dx
        end
    end
    result
end

function r_r_matrix_V(V1, V2, domain, r)
    result = zeros(r, r)
    for j in 1:r
        for l in 1:r
            tjl = 0.0
            for v in axes(V1, 1)
                tjl += V1[v, j] * V2[v, l]
            end
            result[j, l] = tjl * domain.dv
        end
    end
    result
end

function r_r_r_r_tensor(X1, V1, X2, V2, domain, r)
    result = zeros(r, r, r, r)
    for i in 1:r, k in 1:r
        tik = 0
        for x in axes(X1, 1)
            tik += X1[x, i] * X2[x, k]
        end
        tik = tik * domain.dx

        for j in 1:r, l in 1:r
            tjl = 0
            for v in axes(V1, 1)
                tjl += V1[v, j] * V2[v, l]
            end
            tjl = tjl * domain.dv
            result[i, j, k, l] = tik * tjl
        end
    end
    result
end

end


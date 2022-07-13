export step_and_truncate!, truncate!, take_truncated_step!

mutable struct FullTensorSimulation
    domain::Domain
    f::Matrix{Float64}
    E::Vector{Float64}
    η::Vector{Float64}
    q::Float64

    arena::Arena
end

function step!(sim::FullTensorSimulation, Δt)
    @unpack arena, domain, f, E, η = sim
    @unpack nx, dx, dv, x, v, Dx, Dv, Ax, Av = domain

    ρ = sim.q * sum(f, dims=2) .* dv
    E = solve_poisson(Dx, ρ - η)

    df = zeros(size(f))
    
    # Advection step
    for j in eachindex(v)
        A = zeros(1, 1); A[1, 1] = v[j]
        fj = reshape(view(f, :, j), (:, 1))
        df[:, j] = -Δt * flux_limited_linear_hyperbolic(A, fj, Dx, Ax, true, Δt, dx, arena, :free_transport)
    end

    # Force step
    fT = Matrix(f')
    for i in eachindex(x)
        A = zeros(1, 1); A[1, 1] = sim.q * E[i]
        fi = reshape(view(fT, :, i), (:, 1))
        df[i, :] -= Δt * flux_limited_linear_hyperbolic(A, fi, Dv, Av, true, Δt, dv, arena, :lorentz)
    end

    sim.f = f + df
    sim.E = E
end

function step_and_truncate!(sim::FullTensorSimulation, Δt; r)
    step!(sim, Δt)
    truncate!(sim, r=r)
end

function truncate!(sim::FullTensorSimulation; r)
    F = svd(sim.f)
    sim.f = F.U[:, 1:r] * diagm(F.S[1:r]) * F.Vt[1:r, :]
end

function take_truncated_step!(sim::FullTensorSimulation, Δt; r)
    f0 = sim.f
    step!(sim, Δt)
    Δf = sim.f - f0
    F = svd(Δf)
    Δf_r = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    sim.f = f0 + Δf_r
end

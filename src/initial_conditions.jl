using LinearAlgebra

export initial_condition, full_tensor_initial_condition

function initial_condition(domain, f, η; r, q)
    @unpack x, v, dx, dv = domain
    f0 = f.(x, v')
    ρ = sum(f0, dims=2) .* dv

    η0 = η.(x)
    η0 = η0 ./ sum(η0) * sum(ρ)
    @assert sum(ρ - η0) / sum(abs.(ρ)) < 1e-10

    E = solve_poisson(domain.Dx, ρ - η0)

    X, S, V = svd(f0)

    X = X[:, 1:r] / sqrt(dx)
    S = S[1:r] * sqrt(dx * dv)
    V = Matrix(V)[:, 1:r] / sqrt(dv)

    Simulation(domain, X, diagm(S), V, E, η0, q, r, Arena())
end

function full_tensor_initial_condition(domain, f, η; q)
    @unpack x, v, dx, dv = domain
    f0 = f.(x, v')
    ρ = sum(f0, dims=2) .* dv

    η0 = η.(x)
    η0 = η0 ./ sum(η0) * sum(ρ)
    @assert sum(ρ - η0) / sum(abs.(ρ)) < 1e-10

    E = solve_poisson(domain.Dx, ρ - η0)

    FullTensorSimulation(domain, f0, E, η0, q, Arena())
end


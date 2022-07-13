function solve_poisson(D, ρ)
    rhs = vcat(ρ, [0.0])
    # Use a dirichlet bc for ϕ
    L = vcat(-D.D2_centered, spzeros(1, size(D.D2_centered, 1)))
    L[end, 1] = 1.0

    ϕ = L \ rhs
    vec(-D.centered * ϕ)
end

export step_unconventional!

function step_unconventional!(sim::Simulation, Δt)
    @unpack arena, domain, X, S, V, E, η, q, r = sim
    @unpack nx, dx, dv, v, Dx, Dv, Ax, Av = domain

    ρ = q * X * S * sum(V, dims=1)' .* dv
    E = solve_poisson(Dx, ρ - η)

    # K step
    K = alloc_f64!(arena, :K, size(X))
    mul!(K, X, S)
    A¹ = r_r_matrix_V(v .* V, V, domain, r)
    A² = r_r_matrix_V(V, Dv.centered * V, domain, r)

    dK = -Δt * flux_limited_linear_hyperbolic(A¹, K, Dx, Ax, true, Δt, dx, arena, :K)
    KE = alloc_f64!(arena, :KE, size(K))
    @. KE = K * E * q
    dK -= Δt * (KE) * A²'

    K = K + dK
    X1, R1 = qr(K)
    X1 = Matrix(X1) / sqrt(dx)
    R1 = R1 * sqrt(dx)

    @show size(X1')
    @show size(X)
    M = (X1' * X) * dx
    @show size(M)


    # L step
    C¹ = r_r_matrix_X(X, Dx.centered * X, domain, r)
    C² = r_r_matrix_X(X, q * E .* X, domain, r)
    L = (S * V')'
    dL = -Δt * flux_limited_linear_hyperbolic(C², L, Dv, Av, true, Δt, dv, arena, :L)
    dL -= Δt * (L .* v) * C¹'

    L = L + dL
    V1, R̃1 = qr(L)
    V1 = Matrix(V1) / sqrt(dv)
    R̃1 = R̃1 * sqrt(dv)
    N = (V1' * V) * dv

    
    # S step
    S0 = M * S * N
    @show size(S0)

    A¹ = r_r_matrix_V(v .* V1, V1, domain, r)
    A² = r_r_matrix_V(V1, Dv.centered * V1, domain, r)
    C¹ = r_r_matrix_X(X1, Dx.centered * X1, domain, r)
    C² = r_r_matrix_X(X1, q * E .* X1, domain, r)
    @show size(A¹)
    @show size(A²)
    @show size(C¹)
    @show size(C²)
    B¹ = LinearMap(A¹) ⊗ LinearMap(C¹)
    B² = LinearMap(A²) ⊗ LinearMap(C²)
    B = Δt * (B¹ + B²)
    S = S0 + reshape(B * vec(S0), (r, r))

    sim.X = X1
    sim.S = S
    sim.V = V1
    sim.E = E
end

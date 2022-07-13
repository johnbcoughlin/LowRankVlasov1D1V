export step_L_then_K!
function step_L_then_K!(sim::Simulation, Δt; debug=false)
    @unpack arena, domain, X, S, V, E, η, q, r = sim
    @unpack nx, dx, dv, v, Dx, Dv, Ax, Av = domain

    ρ = q * X * S * sum(V, dims=1)' .* dv
    E = solve_poisson(Dx, ρ - η)

    C¹ = r_r_matrix_X(X, Dx.centered * X, domain, r)
    C² = r_r_matrix_X(X, q * E .* X, domain, r)

    # L step
    L = (S * V')'

    dL = -Δt * flux_limited_linear_hyperbolic(C², L, Dv, Av, true, Δt, dv, arena, :L)
    dL -= Δt * (L .* v) * C¹'

    L = L + dL

    V, S = qr(L)
    V = Matrix(V) / sqrt(dv)
    S = S * sqrt(dv)

    f_after_L = X * S * V'
    after_L = alloc_f64!(arena, :after_L, size(f_after_L))
    after_L .= f_after_L

    A¹ = r_r_matrix_V(v .* V, V, domain, r)
    A² = r_r_matrix_V(V, Dv.centered * V, domain, r)

    #B¹ = alloc_f64!(arena, :B¹, (r^2, r^2))
    #kron!(B¹, C¹, A¹)
    B¹ = LinearMap(C¹) ⊗ LinearMap(A¹)
    #B² = alloc_f64!(arena, :B², (r^2, r^2))
    #kron!(B², C², A²)
    B² = LinearMap(C²) ⊗ LinearMap(A²)

    #B = alloc_f64!(arena, :B, (r^2, r^2))
    #@. B = Δt * (B¹ + B²)
    B = Δt * (B¹ + B²)

    # S step
    S = S + reshape(B * vec(S), (r, r))

    f_after_S = X * S * V'
    after_S = alloc_f64!(arena, :after_S, size(f_after_S))
    after_S .= f_after_S

    # K step
    K = alloc_f64!(arena, :K, size(X))
    mul!(K, X, S)
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

    sim.X = X
    sim.S = S
    sim.V = V
    sim.E = E
end

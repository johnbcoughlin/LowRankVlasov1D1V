using Test
using LowRankVlasov1D1V
using LinearAlgebra

function test_is_kind_of_kth_order_in(y, x, k; tol=0.1)
    logx = log.(x)
    logy = log.(y)

    # Test that y is non-increasing
    @test all(diff(logy) .< 0)
    # The overall slope of the convergence should be at most 1
    rise = logy[end] - logy[1]
    run = logx[end] - logx[1]
    @test rise / run > (1 - tol) * k
end

@testset "x derivatives" begin
    Ns = [10, 20, 40, 80, 160, 320]

    left_errors = Float64[]
    left_left_errors = Float64[]
    right_errors = Float64[]
    right_right_errors = Float64[]
    centered_errors = Float64[]

    for nx in Ns
        domain = make_domain(nx=nx, x_min=0.0, x_max=2π, nv=100, v_min=-10.0);
        dx = domain.dx

        u = sin.(2domain.x)
        du = 2cos.(2domain.x)

        Dx = domain.Dx

        du_left = Dx.upwinded_left * u
        du_left_left = Dx.twice_upwinded_left * u
        du_right = Dx.upwinded_right * u
        du_right_right = Dx.twice_upwinded_right * u
        du_centered = Dx.centered * u

        push!(left_errors, norm(du_left - du) * dx)
        push!(right_errors, norm(du_right - du) * dx)
        push!(centered_errors, norm(du_centered - du) * dx)

        push!(left_left_errors, norm(du_left_left - circshift(du, -1)) * dx)
        push!(right_right_errors, norm(du_right_right - circshift(du, 1)) * dx)

        if nx == 10
        end
    end

    test_is_kind_of_kth_order_in(left_errors, 1 ./ Ns, 1)
    test_is_kind_of_kth_order_in(right_errors, 1 ./ Ns, 1)
    test_is_kind_of_kth_order_in(centered_errors, 1 ./ Ns, 2)

    test_is_kind_of_kth_order_in(left_left_errors, 1 ./ Ns, 1)
    test_is_kind_of_kth_order_in(right_right_errors, 1 ./ Ns, 1)
end

@testset "x advection" begin
    Ns = [20, 30, 40]

    errors = Float64[]
    for nx in Ns
        domain = make_domain(nx=nx, x_min=0.0, x_max=2π, nv=10, v_min=-10.0);
        dx = domain.dx
        Dx = domain.Dx
        Ax = domain.Ax

        u = reshape(sin.(2domain.x), (:, 1))
        A = zeros(1, 1); A[1, 1] = 2.4;

        Δt = 0.001
        nt = 400
        for i in 1:nt
            du = -Δt * LowRankVlasov1D1V.flux_limited_linear_hyperbolic(A, u, Dx, Ax, true, Δt, dx)
            u += du
        end
        T = Δt * nt

        u_expected = sin.(2(domain.x .- 2.4*T))
        error = norm(u - u_expected) * domain.dx
        push!(errors, error)
    end

    @show errors
end

@testset "v advection" begin
    Ns = [60]

    errors = Float64[]
    for nv in Ns
        domain = make_domain(nx=10, x_min=0.0, x_max=2π, nv=nv, v_min=-10.0);
        dv = domain.dv
        Dv = domain.Dv
        Av = domain.Av
        v = domain.v

        f = reshape(exp.(-v.^2 ./ 2), (:, 1))
        E = zeros(1, 1); E[1, 1] = 2.4;

        Δt = 0.004
        nt = 400
        for i in 1:nt
            df = -Δt * LowRankVlasov1D1V.flux_limited_linear_hyperbolic(E, f, Dv, Av, true, Δt, dv)
            f += df
        end
        T = Δt * nt

        display(f)

        f_expected = exp.(-(v .- E[1, 1]*nt*Δt).^2 ./ 2)
        display(f_expected)

        display(f - f_expected)
        error = norm(f - f_expected) * domain.dv
        push!(errors, error)
    end

end

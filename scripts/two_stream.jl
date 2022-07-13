using DrWatson
@quickactivate :LowRankVlasov1D1V


domain = make_domain(nx=100, x_min=0.0, x_max=2π, nv=100, v_min=-10.0);

f0(x, v) = begin
    ρ(x) = 1.0 + 0.0 * sin(x)
    M(v, u, T) = 1 / (√(2π*T)) * exp(-(v - u)^2 / 2T)

    ρ(x) * (M(v, 2.0, 0.5) + M(v, -2.0, 0.5))
end

@show size(domain.x)
η0(x) = 1.0

sim = initial_condition(domain, f0, η0, r=8)

f_init = expand_f(sim)

t = 0.0
Δt = 0.001
while t < 0.02
    @show t
    step!(sim, Δt)
    global t += Δt
end


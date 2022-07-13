"""
    Compute the time derivative due to the flux with Jacobian A
"""
function flux_limited_linear_hyperbolic(A, U, D, average, periodic, Δt, d, arena, sym)
    r = size(A, 1)
    @assert size(A) == (r, r)
    @assert size(U, 2) == r
    @assert A ≈ A'

    dÛ = zeros(size(U))

    # By symmetry of A, see above, imaginary parts are negligible.
    Λ, R = eigen(Symmetric(A))
    Λ = real.(Λ)
    R = real.(R)

    Û = alloc_f64!(arena, Symbol(sym, '_', :Û), size(U))
    mul!(Û, U, R)

    for j in 1:r
        λ = Λ[j]
        κ = λ * Δt / d
        if κ > 0.8
            @warn "CFL number exceeding 0.8"
        end

        dÛ[:, j] = limited_flux_difference(λ, view(Û, :, j), D, average, Δt, d, arena, sym)
    end

    dU = alloc_f64!(arena, Symbol(sym, '_', :dU), size(dÛ))
    mul!(dU, dÛ, R')

    dU
end

function limited_flux_difference(λ, U, D, average, Δt, dx, arena, sym)
    left_firstorder = alloc_f64!(arena, Symbol(sym, '_', :left_firstorder), size(U))
    # left_firstorder = λ * (average.upwinded_left * U) - 0.5 * abs(λ) * (D.upwinded_left * U)
    mul!(left_firstorder, D.upwinded_left, U, - 0.5 * abs(λ), 0.0)
    mul!(left_firstorder, average.upwinded_left, U, λ, 1.0)

    
    right_firstorder = alloc_f64!(arena, Symbol(sym, '_', :right_firstorder), size(U))
    # right_firstorder = λ * (average.upwinded_right * U) - 0.5 * abs(λ) * (D.upwinded_right * U)
    mul!(right_firstorder, D.upwinded_right, U, - 0.5 * abs(λ), 0.0)
    mul!(right_firstorder, average.upwinded_right, U, λ, 1.0)

    θleft = limiter_θ(U, D, :left, λ, arena, sym)
    θright = limiter_θ(U, D, :right, λ, arena, sym)

    F_L = right_firstorder - left_firstorder

    ν = λ * Δt / dx

    correction_left = alloc_f64!(arena, Symbol(sym, '_', :correction_left), size(U))
    mul!(correction_left, D.upwinded_left, U, 0.5 * (sign(ν) - ν) * λ, 0.0)
    correction_right = alloc_f64!(arena, Symbol(sym, '_', :correction_right), size(U))
    mul!(correction_right, D.upwinded_right, U, 0.5 * (sign(ν) - ν) * λ, 0.0)

    vanleer(θ) = θ == Inf ? 2.0 : (abs(θ) + θ) / (1 + abs(θ))


    correction = vanleer.(θright) .* correction_right - vanleer.(θleft) .* correction_left

    return F_L + 1.0*correction
end

function limiter_θ(U, D, interface, λ, arena, sym)
    if interface == :left
        numerator = alloc_f64!(arena, Symbol(sym, '_', :numerator_left), size(U))
        denominator = alloc_f64!(arena, Symbol(sym, '_', :denominator_left), size(U))

        mul!(denominator, D.upwinded_left, U)
        if λ > 0
            mul!(numerator, D.twice_upwinded_left, U)
        else
            mul!(numerator, D.upwinded_right, U)
        end
    elseif interface == :right
        numerator = alloc_f64!(arena, Symbol(sym, '_', :numerator_right), size(U))
        denominator = alloc_f64!(arena, Symbol(sym, '_', :denominator_right), size(U))

        mul!(denominator, D.upwinded_right, U)
        if λ > 0
            mul!(numerator, D.upwinded_left, U)
        else
            mul!(numerator, D.twice_upwinded_right, U)
        end
    end

    θ(num, denom) = begin
        if denom == 0 && num == 0
            return 0.0
        elseif denom == 0
            return Inf
        else
            return num / denom
        end
    end
    if interface == :left
        result = alloc_f64!(arena, Symbol(sym, '_', :θ_left), size(U))
        @. result = θ(numerator, denominator)
    elseif interface == :right
        result = alloc_f64!(arena, Symbol(sym, '_', :θ_right), size(U))
        @. result = θ(numerator, denominator)
    else
        error("invalid interface")
    end
end

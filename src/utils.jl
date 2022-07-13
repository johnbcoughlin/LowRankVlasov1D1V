using SparseArrays

export make_domain

struct Derivatives
    upwinded_left::SparseMatrixCSC{Float64, Int}
    twice_upwinded_left::SparseMatrixCSC{Float64, Int}

    upwinded_right::SparseMatrixCSC{Float64, Int}
    twice_upwinded_right::SparseMatrixCSC{Float64, Int}

    centered::SparseMatrixCSC{Float64, Int}

    D2_centered::SparseMatrixCSC{Float64, Int}
end

# Operators that compute the average divided by the grid spacing dx
struct Averages
    upwinded_left::SparseMatrixCSC{Float64, Int}
    upwinded_right::SparseMatrixCSC{Float64, Int}
end

# Periodic domain in x, non-periodic in v
function make_domain(; nx, x_min, x_max, nv, v_min, v_max=-v_min)
    Lx = x_max - x_min
    dx = Lx / nx

    x = x_min:dx:(x_max-dx)
    @assert length(x) == nx

    Lv = v_max - v_min
    dv = Lv / (nv-1)

    v = v_min:dv:v_max
    @assert length(v) == nv

    Dx = derivatives(nx, dx, true)
    Dv = derivatives(nv, dv, false)

    Ax = averages(nx, dx, true)
    Av = averages(nv, dv, false)

    Domain(nx, dx, x_min, x_max, x, Dx, Ax,
           nv, dv, v_min, v_max, v, Dv, Av)
end

derivatives(n, d, periodic) = begin
    D_left = spdiagm(-1 => -ones(n-1), 0 => ones(n))
    D_left_left = spdiagm(-2 => -ones(n-2), -1 => ones(n-1))
    D_right = spdiagm(0 => -ones(n), 1 => ones(n-1))
    D_right_right = spdiagm(1 => -ones(n-1), 2 => ones(n-2))
    D_centered = spdiagm(-1 => -ones(n-1), 1 => ones(n-1))

    D2_centered = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))

    if periodic
        D_left[1, end] = -1
        
        D_left_left[1, end-1:end] = [-1, 1]
        D_left_left[2, end] = -1

        D_right[end, 1] = 1

        D_right_right[end, 1:2] = [-1, 1]
        D_right_right[end-1, 1] = 1

        D_centered[1, end] = -1
        D_centered[end, 1] = 1

        D2_centered[1, end] = D2_centered[end, 1] = 1
    else
        # Copy out boundary conditions
        D_left[1, 1] = 0
        D_left_left[2, 1] = 0

        D_right[end, end] = 0
        D_right_right[end-1, end] = 0

        D_centered[1, 1:2] = [-1, 1]
        D_centered[end, end-1:end] = [-1, -1]
    end

    Derivatives(D_left ./ d, 
                D_left_left ./ d, 
                D_right ./ d, 
                D_right_right ./ d, 
                D_centered ./ 2d, 
                D2_centered ./ d^2)
end

averages(n, d, periodic) = begin
    C_left = spdiagm(-1 => ones(n-1), 0 => ones(n))
    C_right = spdiagm(0 => ones(n), 1 => ones(n-1))

    if periodic
        C_left[1, end] = 1
        C_right[end, 1] = 1
    else
        C_left[1, 1] = 2
        C_right[end, end] = 2
    end

    Averages(C_left ./ 2d, C_right ./ 2d)
end

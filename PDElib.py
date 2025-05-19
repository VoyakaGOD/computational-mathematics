from typing import Callable
from dataclasses import dataclass
from ODElib import LinearSolver, Vector

RtoR = Callable[[float], float]
R2toR = Callable[[float, float], float]

@dataclass
class Grid:
    # time:
    T : float
    N : int
    # spatial:
    L : float
    M : int

# Convection Equation Solver
# solve u't + c*u'x = f(t, x)
# WARNING: c > 0
# u(0, x) = phi(x)
# u(t, 0) = psi(t)
class CESolver:
    @staticmethod
    def __prepare(grid : Grid, phi : RtoR, psi : RtoR):
        tau = grid.T / grid.N
        h = grid.L / grid.M
        u = [[phi(m*h) for m in range(grid.M + 1)]]
        for n in range(1, grid.N + 1):
            u += [[psi(tau*n)] + [0 for _ in range(grid.M)]]
        return tau, h, u

    @staticmethod
    def solve_by_upwind_scheme(c : float, grid : Grid, phi : RtoR, psi : RtoR, f : R2toR):
        tau, h, u = CESolver.__prepare(grid, phi, psi)
        cn = c * tau / h # Courant number
        for n in range(grid.N):
            for m in range(1, grid.M + 1):
                u[n+1][m] = u[n][m] - cn*(u[n][m] - u[n][m-1]) + tau*f(tau * n, h * m)
        return u

    @staticmethod
    def solve_by_rectangle_scheme(c : float, grid : Grid, phi : RtoR, psi : RtoR, f : R2toR):
        tau, h, u = CESolver.__prepare(grid, phi, psi)
        cn = c * tau / h # Courant number
        ct = 1 / (1 + cn)
        ch = cn * ct
        cf = 2 * tau / (1 + cn)
        for n in range(grid.N):
            for m in range(1, grid.M + 1):
                f_nm = f(tau * (n + 0.5), h * (m - 0.5))
                u[n+1][m] = ct*(u[n][m] + u[n][m-1] - u[n+1][m-1]) - ch*(u[n][m] - u[n][m-1] - u[n+1][m-1]) + cf*f_nm
        return u

    @staticmethod
    def solve_by_Lax_Wendroff_scheme(c: float, grid: Grid, phi: RtoR, psi: RtoR, f: R2toR):
        tau, h, u = CESolver.__prepare(grid, phi, psi)
        cn = c * tau / h  # Courant number
        c1 = cn / 2
        c2 = cn*cn / 2
        for n in range(grid.N):
            for m in range(1, grid.M):
                u[n+1][m] = u[n][m] - c1*(u[n][m+1] - u[n][m-1]) + c2*(u[n][m+1] - 2*u[n][m] + u[n][m-1]) + tau * f(tau * n, h * m)
            # upwind on border, may reduce order to 1st...
            u[n+1][grid.M] = u[n][grid.M] - cn*(u[n][grid.M] - u[n][grid.M-1]) + tau*f(tau * n, h * grid.M)
        return u

@dataclass
class HEParams:
    rho : float # density
    T_p : float # phase change temperature
    L : float # heat of fusion
    # fluid:
    c_f : float # heat capacity
    a_f : float # thermal conductivity
    #solid:
    c_s : float # heat capacity
    a_s : float # thermal conductivity

# Heat Equation Solver
# solve rho*c*T't = a*T''x = 0
# c = c_s if x < y(t) else c = c_f
# a = a_s if x < y(t) else a = a_f
# y(t) defined by rule T(y(t), t) = T_p
# WARNING: The temperature should increase monotonically from left to right!
# takes into account the balance of energies:
# a_s*T'x(y(t) - 0) - a_f*T'x(y(t) + 0) = rho*L*y't
# T(x, 0) = phi(1)
# T(0, t) = psi_l(t)
# T(grid.L, t) = psi_r(t)
# returns (y[t], T[t][x])
class HESolver:
    T_eps = 0.1 # for energy balance

    @staticmethod
    def __prepare(grid : Grid, phi : RtoR, psi_l : RtoR, psi_r : RtoR):
        tau = grid.T / grid.N
        h = grid.L / grid.M
        u = [[phi(m*h) for m in range(grid.M + 1)]]
        for n in range(1, grid.N + 1):
            u += [[psi_l(tau*n)] + [0 for _ in range(grid.M - 1)] + [psi_r(tau*n)]]
        return tau, h, u

    # predicts next position of phase front with first order
    # but using continuous front position and correction of values
    @staticmethod
    def __predict_phase_front(params: HEParams, grid: Grid, T: list[float], y: list[float]):
        h = grid.L / grid.M
        tau = grid.T / grid.N
        current_y = y[-1]
        m = min(max(int(current_y / h), 1), grid.M - 2)
        if current_y > 0.0:
            dTdx_left = (params.T_p - T[m-1]) / (current_y - (m-1)*h)
        else:
            dTdx_left = 0.0
        if m < grid.M - 1:
            dTdx_right = (T[m+1] - params.T_p) / ((m+1)*h - current_y)
        else:
            dTdx_right = 0.0
        dy_dt = (params.a_s * dTdx_left - params.a_f * dTdx_right) / (params.rho * params.L)
        new_y = current_y + dy_dt * tau
        new_y = max(0.0, min(new_y, grid.L))
        m_new = min(max(int(new_y / h), 0), grid.M - 1)
        xi_new = (new_y - m_new * h) / h
        T_at_new_y = (1 - xi_new) * T[m_new] + xi_new * T[m_new + 1] if m_new < grid.M else T[m_new]
        # move y(t) if |T(y(t), t) - T_p| > T_eps:
        if (abs(T_at_new_y - params.T_p) > HESolver.T_eps) and (m_new < grid.M - 1):
            if T[m_new] != T[m_new + 1]:
                xi_corrected = (params.T_p - T[m_new]) / (T[m_new + 1] - T[m_new])
                new_y = h * (m_new + xi_corrected)
        return new_y

    @staticmethod
    def __solve_explicit(params : HEParams, grid: Grid, phi: RtoR, psi_l: RtoR, psi_r: RtoR, __scheme):
        tau, h, u = HESolver.__prepare(grid, phi, psi_l, psi_r)
        k_s = params.a_s / (params.rho * params.c_s) * tau/h/h
        k_f = params.a_f / (params.rho * params.c_f) * tau/h/h
        y = [0]
        for n in range(grid.N):
            y += [HESolver.__predict_phase_front(params, grid, u[n], y)]
            boundary = min(max(int(y[-1] / h), 1), grid.M)
            for m in range(1, boundary):
                u[n+1][m] = __scheme(k_s, u, n, m)
            for m in range(boundary, grid.M):
                u[n+1][m] = __scheme(k_f, u, n, m)
        return y, u
    
    @staticmethod
    def __solve_implicit(params : HEParams, grid: Grid, phi: RtoR, psi_l: RtoR, psi_r: RtoR, __scheme):
        h = grid.L / grid.M
        tau = grid.T / grid.N
        u = [[phi(m*h) for m in range(grid.M + 1)]]
        k_s = params.a_s / (params.rho * params.c_s) * tau/h/h
        k_f = params.a_f / (params.rho * params.c_f) * tau/h/h
        y = [0]
        A = [0 for _ in range(grid.M - 1)]
        B = [0 for _ in range(grid.M - 1)]
        C = [0 for _ in range(grid.M - 1)]
        f = [0 for _ in range(grid.M - 1)]
        for n in range(grid.N):
            y += [HESolver.__predict_phase_front(params, grid, u[n], y)]
            boundary = min(max(int(y[-1] / h), 0), grid.M - 1)
            for m in range(boundary):
                A[m], B[m], C[m], f[m] = __scheme(k_s, u, n, m + 1)
            for m in range(boundary, grid.M - 1):
                A[m], B[m], C[m], f[m] = __scheme(k_f, u, n, m + 1)
            t = tau * (n + 1)
            u += [[psi_l(t)] + LinearSolver.solve_tridiagonal(A, B, C, Vector(f)).toList() + [psi_r(t)]]
        return y, u

    @staticmethod
    def solve_by_explicit_4point_scheme(params : HEParams, grid: Grid, phi: RtoR, psi_l: RtoR, psi_r: RtoR):
        scheme = lambda k, u, n, m: u[n][m] + k * (u[n][m+1] - 2*u[n][m] + u[n][m-1])
        return HESolver.__solve_explicit(params, grid, phi, psi_l, psi_r, scheme)

    @staticmethod
    def solve_by_implicit_4point_scheme(params : HEParams, grid: Grid, phi: RtoR, psi_l: RtoR, psi_r: RtoR):
        tau = grid.T / grid.N
        def scheme(k, u, n, m):
            f = u[n][m]
            t = tau*(n+1)
            if m == 1:
                f += k*psi_l(t)
            elif m == (grid.M - 1):
                f += k*psi_r(t)
            return -k, 1 + 2*k, -k, f
        return HESolver.__solve_implicit(params, grid, phi, psi_l, psi_r, scheme)

    @staticmethod
    def solve_by_Crank_Nicolson_scheme(params : HEParams, grid: Grid, phi: RtoR, psi_l: RtoR, psi_r: RtoR):
        tau = grid.T / grid.N
        def scheme(k, u, n, m):
            r = k/2
            f = r*u[n][m+1] + (1-k)*u[n][m] + r*u[n][m-1]
            t = tau*(n+1)
            if m == 1:
                f += r*psi_l(t)
            elif m == (grid.M - 1):
                f += r*psi_r(t)
            return -r, 1 + k, -r, f
        return HESolver.__solve_implicit(params, grid, phi, psi_l, psi_r, scheme)

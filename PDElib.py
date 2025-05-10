from typing import Callable
from dataclasses import dataclass
RtoR = Callable[[float], float]
R2toR = Callable[[float, float], float]

@dataclass
class Grid:
    # time
    T : float
    N : int
    # spatial
    L : float
    M : int

# solve u't + c*u'x = f(t, x)
# WARNING: c > 0
# u(0, x) = phi(x)
# u(t, 0) = psi(t)
class CESolver:
    def __prepare(grid : Grid, phi : RtoR, psi : RtoR):
        tau = grid.T / grid.N
        h = grid.L / grid.M
        u = [[phi(m*h) for m in range(grid.M + 1)]]
        for n in range(1, grid.N + 1):
            u += [[psi(tau*n)] + [0 for _ in range(grid.M)]]
        return tau, h, u

    def solve_by_upwind_scheme(c : float, grid : Grid, phi : RtoR, psi : RtoR, f : R2toR):
        tau, h, u = CESolver.__prepare(grid, phi, psi)
        cn = c * tau / h # Courant number
        for n in range(grid.N):
            for m in range(1, grid.M + 1):
                u[n+1][m] = u[n][m] - cn*(u[n][m] - u[n][m-1]) + tau*f(tau * n, h * m)
        return u

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

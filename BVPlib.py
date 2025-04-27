from ODElib import *

FuncT2D = Callable[[float, float, float], float]

# y'' = f(x, y, y')
# y(x1) = y1
# y(x1 + Nh) = y2
class BVPSolver:
    eps = 1e-9
    step = 1e-7

    # u'' + A(x)u' + B(x)u = 0
    # u0 = 0, u0' = 1
    @staticmethod
    def __solve_linear(A : list[float], B : list[float], h : float, N : int):
        u = [0.0, (2 + A[1]*h) / (2 - B[1]*h*h) * h, 2 * h]
        for n in range(2, N):
            u += [(2 * (2 - B[n]*h*h)*u[n] - (2 - A[n]*h)*u[n-1]) / (2 + A[n]*h)]
        return u

    @staticmethod
    def solve_by_shooting_method(f : FuncT2D, h : float, N: int, y1 : float, y2 : float, x1 : float = 0):
        g : VecTFunc = lambda x, q: Vector(q[1], f(x, q[0], q[1]))
        alpha = 0
        step = BVPSolver.step
        while(True):
            probe = RungeKuttaMethods.explicit_4_order(g, Vector(y1, alpha), h, N, x1)
            boundary = probe[N][0] - y2
            ODEIOSettings.info("Boundary vlaue: " + str(probe[N][0]))
            if(abs(boundary) < BVPSolver.eps):
                break
            A = [(f(n*h, probe[n][0] - step, probe[n][1]) - f(n*h, probe[n][0] + step, probe[n][1])) / (2 * step) for n in range(N)]
            B = [(f(n*h, probe[n][0], probe[n][1] - step) - f(n*h, probe[n][0], probe[n][1] + step)) / (2 * step) for n in range(N)]
            alpha = alpha - boundary / BVPSolver.__solve_linear(A, B, h, N)[N]
        return RungeKuttaMethods.explicit_4_order(g, Vector(y1, alpha), h, N, x1)
    
    # u'' + p(x)u' + q(x)u + r(x) = 0
    # u0 = 0, uN = 0
    @staticmethod
    def __solve_linear_BVP(p : list[float], q : list[float], r : list[float], h : float, N : int):
        As = [1/h/h + p[n]/(2*h) for n in range(1, N)]
        Bs = [-2/h/h + q[n] for n in range(1, N)]
        Cs = [1/h/h - p[n]/(2*h) for n in range(1, N)]
        ys = [0] + LinearSolver.solve_tridiagonal(As, Bs, Cs, Vector(r[1:-1])).toList() + [0]
        us = [(ys[n], (ys[n+1] - ys[n-1]) / (2*h)) for n in range(1, N)]
        us = [(ys[0], (4*ys[1] - ys[2]) / (2*h))] + us + [(ys[N], (ys[N-2] - 4*ys[N-1]) / (2*h))]
        us = [Vector(u[0], u[1], r[n] - p[n]*u[1] - q[n]*u[0]) for n, u in enumerate(us)]
        return us
    
    @staticmethod
    def solve_by_quasilinearization_method(f : FuncT2D, h : float, N: int, y1 : float, y2 : float, x1 : float = 0) -> list[Vector]:
        x2 = x1 + N*h
        step = BVPSolver.step
        ly = [Vector((n*h - x2)/(x1 - x2) * y1 + (n*h - x1)/(x2 - x1) * y2, (y2 - y1) / (N*h), 0) for n in range(N + 1)]
        while(True):
            p = [(f(n*h, ly[n][0], ly[n][1] - step) - f(n*h, ly[n][0], ly[n][1] + step)) / (2 * step) for n in range(N + 1)]
            q = [(f(n*h, ly[n][0] - step, ly[n][1]) - f(n*h, ly[n][0] + step, ly[n][1])) / (2 * step) for n in range(N + 1)]
            r = [f(n*h, ly[n][0], ly[n][1]) - ly[n][2] for n in range(N + 1)]
            nu = BVPSolver.__solve_linear_BVP(p, q, r, h, N)
            ly = [ly[n] + nu[n] for n in range(N + 1)]
            if (max([abs(nu[n][0]) for n in range(N + 1)]) < BVPSolver.eps):
                break
        return ly

from ODElib import *

class SLPSolver:
    eps = 1e-6

    # (S(x)y')' + lb*S(x)y = 0
    # y'(0) = 0, R(y, y') = 0
    # returns lb, (y, y')_n
    @staticmethod
    def solve_by_shooting_method(S : RtoR, R : R2toR, N: int, L : float, initial_lambda : float):
        lb = initial_lambda
        h = L/N
        def M(l : float):
            f : VecTFunc = lambda x, q: Vector(q[1], -l * q[0] - diff(S, x)/S(x) * q[1])
            fL = RungeKuttaMethods.explicit_4_order(f, Vector(1, 0), h, N, 0)[N]
            return R(fL[0], fL[1])
        while(True):
            ODEIOSettings.info("lb = " + str(lb))
            boundary = M(lb)
            if(abs(boundary) < SLPSolver.eps):
                break
            lb = lb - boundary / diff(M, lb)
        f : VecTFunc = lambda x, q: Vector(q[1], -lb * q[0] - diff(S, x)/S(x) * q[1])
        return lb, RungeKuttaMethods.explicit_4_order(f, Vector(1, 0), h, N, 0)

    # (S(x)y')' + lb*S(x)y = 0
    # y'(0) = 0, c1*y'(L) + c0*y(L) = 0
    # returns lb, y_n
    @staticmethod
    def solve_by_augmented_vector_method(S : RtoR, c1 : float, c0 : float, N: int, L : float, initial_lambda : float) -> tuple[float, list[float]]:
        u = Vector([0, 0, 1] + [0] * (N - 2) + [initial_lambda])
        h = L / N
        def get_F_at(n : int, u : Vector):
            return S((n + 0.5)*h)*(u[n+1] - u[n]) + S((n - 0.5)*h)*(u[n-1] - u[n]) + u[-1]*h*h*S(n*h)*u[n]
        while True:
            ODEIOSettings.info("lb = " + str(u[-1]))
            F = Vector([4*u[1] - 3*u[0] - u[2]] + [get_F_at(n, u) for n in range(1, N)] + [c1*(3*u[N] - 4*u[N-1] + u[N-2]) + 2*h*c0*u[N]])
            b = Vector([h*h*S(h)/S(3/2*h)] + [h*h*S(n*h)*u[n] for n in range(1, N)] + [-h*h*S((N-1)*h)/S((N-3/2)*h)])
            A = [0] + [S((n - 0.5)*h) for n in range(1, N)] + [-4*c1]
            B = [-3] + [u[-1]*h*h*S(n*h) - S((n - 0.5)*h) - S((n + 0.5)*h) for n in range(1, N)] + [3*c1 + 2*h*c0]
            C = [4] + [S((n + 0.5)*h) for n in range(1, N)]
            f0 = 1/S(3/2*h)
            B[0] -= f0 * A[1]
            C[0] -= f0 * B[1]
            fN = c1/S((N-3/2)*h)
            A[-1] -= fN * B[-2]
            B[-1] -= fN * C[-2]
            p = LinearSolver.solve_tridiagonal(A, B, C, -F)
            q = LinearSolver.solve_tridiagonal(A, B, C, -b)
            lb = (1 - u[2] - p[2]) / q[2]
            du = Matrix((p + lb*q).data + [[lb]])
            u += du
            if NonlinearSolver.norm(du) < NonlinearSolver.eps:
                break
        return u[N + 1], Vector(u).toList()[:-1]

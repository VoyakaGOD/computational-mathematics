from typing import Callable
from matrixlib import *

VecTFunc = Callable[[float, Vector], Vector]
VecFunc = Callable[[Vector], Vector]

class ODEIOSettings:
    out : Callable[[float], None] = lambda percentage: None

class LinearSolver:
    norm = get_Euclidian_norm
    eps = 1e-9

    # solve Ax = f by Seidel, with initial approximation of x = x0
    @staticmethod
    def solve_system(A : Matrix, f : Vector, x0 : Vector) -> Vector:
        f = A.T() * f
        A = A.T() * A
        n = A.n
        x = x0.copy()
        x_dual = Vector.zeros(n)
        while LinearSolver.norm(A * x - f) > LinearSolver.eps:
            for i in range(n):
                x_dual[i] = -(sum([A[i, j] * x_dual[j] for j in range(i) if i != j]) 
                            + sum([A[i, j] * x[j] for j in range(i+1, n) if i != j]) 
                            - f[i]) / A[i, i]
            x, x_dual = x_dual, x
        return x

    # solve system in form A_i*x_{i-1} + B_i*x_i + C_i*x_{i+1} = f_i
    @staticmethod
    def solve_tridiagonal(As : tuple[float], Bs : tuple[float], Cs : tuple[float], f : Vector):
        a = [0, -Cs[0] / Bs[0]]
        b = [0, f[0] / Bs[0]]
        n = f.m - 1
        for i in range(1, n):
            denominator = a[i]*As[i] + Bs[i]
            a += [-Cs[i] / denominator]
            b += [(f[i] - As[i]*b[i]) / denominator]
        x = Vector.zeros(n + 1)
        x[n] = (f[n] - b[n]*As[n]) / (Bs[n] + a[n]*As[n])
        for i in range(n, 0, -1):
            x[i-1] = a[i]*x[i] + b[i]
        return x

    # solve system in form A_i*x_{i-1} + B_i*x_i + C_i*x_{i+1} = f_i, 0 < i < (n-1)
    # B_0*x_0 + C_1*x_1 + A_0*x_{n-1} = f_0
    # C_{n-1}*x_0 + A_{n-1}*x_{n-2} + B_{n-1}*x_{n-1} = f_{n-1}
    @staticmethod
    def solve_cyclic_tridiagonal(As : tuple[float], Bs : tuple[float], Cs : tuple[float], f : Vector):
        a = [0, -Cs[0] / Bs[0]]
        b = [0, f[0] / Bs[0]]
        c = [0, -As[0] / Bs[0]]
        n = f.m - 1
        for i in range(1, n):
            denominator = a[i]*As[i] + Bs[i]
            a += [-Cs[i] / denominator]
            b += [(f[i] - As[i]*b[i]) / denominator]
            c += [-As[i]*c[i] / denominator]
        denominator = Bs[n] + As[n]*(a[n] + c[n])
        mu = [-Cs[n] / denominator]
        nu = [(f[n] - As[n]*b[n]) / denominator]
        for i in range(n):
            n_i = n - i
            mu += [a[n_i]*mu[i] + c[n_i]*mu[0]]
            nu += [a[n_i]*nu[i] + b[n_i] + c[n_i]*nu[0]]
        x = Vector.zeros(n + 1)
        x[0] = nu[n] / (1 - mu[n])
        for i in range(n):
            x[n-i] = mu[i]*x[0] + nu[i]
        return x

# calculate Jacobian with 2nd order method
class Differetiator:
    step = 1e-9

    @staticmethod
    def J(F : VecFunc, x : Vector):
        m = x.m
        dxs = [Vector([Differetiator.step if k == n else 0 for k in range(m)]) for n in range(m)]
        dfs = [(F(x + dxs[n]) - F(x - dxs[n])) / (2 * Differetiator.step) for n in range(m)]
        data = [[dfs[n][k] for n in range(m)] for k in range(m)]
        return Matrix(data)

# solve F(x) = 0 by Newton, with initial approximation of x = x0
class NonlinearSolver:
    norm = get_Euclidian_norm
    eps = 1e-9

    @staticmethod
    def solve_system(F : VecFunc, x0 : Vector):
        x = x0.copy()
        while True:
            dx = LinearSolver.solve_system(Differetiator.J(F, x), -F(x), Vector.zeros(x.m))
            x += dx
            if NonlinearSolver.norm(dx) < NonlinearSolver.eps:
                break
        return x

# t_n = t_0 + nh, n in [0, N]
class RungeKuttaMethods:
    @staticmethod
    def explicit_1_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            u += [u[n] + h * f(t, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_2_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h / 2, u[n] + h / 2 * k1)
            u += [u[n] + h * k2]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_3_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h/3, u[n] + h * k1/3)
            k3 = f(t + 2/3 * h, u[n] + 2/3 * h * k2)
            u += [u[n] + h * 0.25 * k1 + h * 0.75 * k3]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_4_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h/2, u[n] + h * k1/2)
            k3 = f(t + h/2, u[n] + h * k2/2)
            k4 = f(t + h, u[n] + h * k3)
            u += [u[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]
            ODEIOSettings.out((n+1) / N)
        return u
    
    @staticmethod
    def implicit_1_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            F = lambda x: x - h*f(t + h, u[n] + x)
            k1 = NonlinearSolver.solve_system(F, Vector.zeros(u0.m))
            u += [u[n] + k1]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_2_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            F = lambda x: x - h*f(t + h/2, u[n] + x/2)
            k1 = NonlinearSolver.solve_system(F, Vector.zeros(u0.m))
            u += [u[n] + k1]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_3_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            g = 0.7887
            F1 = lambda x: x - h*f(t + g*h, u[n] + g*x)
            k1 = NonlinearSolver.solve_system(F1, Vector.zeros(u0.m))
            F2 = lambda x: x - h*f(t + (1-g)*h, u[n] + (1-2*g)*k1 + g*x)
            k2 = NonlinearSolver.solve_system(F2, Vector.zeros(u0.m))
            u += [u[n] + (k1 + k2)/2]
            ODEIOSettings.out((n+1) / N)
        return u
    
    @staticmethod
    def implicit_4_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            F1 = lambda x: x - h*f(t + h/2, u[n] + x/2)
            k1 = NonlinearSolver.solve_system(F1, Vector.zeros(u0.m))
            F2 = lambda x: x - h*f(t + 2/3*h, u[n] + 2/3*x)
            k2 = NonlinearSolver.solve_system(F2, Vector.zeros(u0.m))
            F3 = lambda x: x - h*f(t + 1/2*h, u[n] - 5/2*k1 + 5/2*k2 + 1/2*x)
            k3 = NonlinearSolver.solve_system(F3, Vector.zeros(u0.m))
            F4 = lambda x: x - h*f(t + 1/3*h, u[n] - 5/3*k1 + 4/3*k2 + 2/3*x)
            k4 = NonlinearSolver.solve_system(F4, Vector.zeros(u0.m))
            u += [u[n] - k1 + 3/2*k2 - k3 + 3/2*k4]
            ODEIOSettings.out((n+1) / N)
        return u

# u0 = [u_0, ..., u_p], p - order of method
class AdamsMethods:
    @staticmethod
    def explicit_1_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        return RungeKuttaMethods.explicit_1_order(f, u0[0], h, N, t0)
    
    @staticmethod
    def explicit_2_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            u += [u[n] + h/2 * (3*f(t, u[n]) - f(t-h, u[n-1]))]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [u[n] + h/12 * (23*f(t, u[n]) - 16*f(t-h, u[n-1]) + 5*f(t-2*h, u[n-2]))]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [u[n] + h/24 * (55*f(t, u[n]) - 59*f(t-h, u[n-1]) + 37*f(t-2*h, u[n-2]) - 9*f(t-3*h, u[n-3]))]
            ODEIOSettings.out((n+1) / N)
        return u
    
    @staticmethod
    def implicit_1_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(N):
            t = t0 + n * h
            F = lambda x: x - h * f(t + h, x) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_2_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            F = lambda x: x - h/2 * (f(t + h, x) + f(t, u[n])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            F = lambda x: x - h/12 * (5*f(t + h, x) + 8*f(t, u[n]) - f(t-h, u[n-1])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u
    
    @staticmethod
    def implicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            F = lambda x: x - h/24 * (9*f(t + h, x) + 19*f(t, u[n]) - 5*f(t-h, u[n-1]) + f(t-2*h, u[n-2])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

# Backward Differentiation Formulas
# u0 = [u_0, ..., u_p], p - order of method
class BDF:
    @staticmethod
    def explicit_1_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        return AdamsMethods.explicit_1_order(f, u0, h, N, t0)

    @staticmethod
    def explicit_2_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            u += [u[n-1] + 2*h * f(t, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [-3/2*u[n] + 3*u[n-1] - 1/2*u[n-2] + 3*h * f(t, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def explicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [-10/3*u[n] + 6*u[n-1] - 2*u[n-2] + 1/3*u[n-3] + 4*h * f(t, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_1_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        return AdamsMethods.implicit_1_order(f, u0, h, N, t0)

    @staticmethod
    def implicit_2_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            F = lambda x: 3/2*x - 2*u[n] + 1/2*u[n-1] - h * f(t+h, x)
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            F = lambda x: 11/6*x - 3*u[n] + 3/2*u[n-1] - 1/3*u[n-2] - h * f(t+h, x)
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            F = lambda x: 25/12*x - 4*u[n] + 3*u[n-1] - 4/3*u[n-2] + 1/4*u[n-3] - h * f(t+h, x)
            u += [NonlinearSolver.solve_system(F, u[n])]
            ODEIOSettings.out((n+1) / N)
        return u

# for autonomus systems
class RosenbrockWannerMethods:
    @staticmethod
    def implicit_1_order(f : VecFunc, u0 : Vector, h : float, N : int):
        u = [u0]
        J = Differetiator.J(f, u0)
        dim = u0.m
        A = Matrix.identity(dim) - h  * J
        for n in range(N):
            k1 = LinearSolver.solve_system(A, h * f(u[n]), Vector.zeros(dim))
            u += [u[n] + k1]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_2_order(f : VecFunc, u0 : Vector, h : float, N : int):
        u = [u0]
        g = 1.7071
        b = 0.2929
        J = Differetiator.J(f, u0)
        dim = u0.m
        A = Matrix.identity(dim) - h * g  * J
        for n in range(N):
            k1 = LinearSolver.solve_system(A, h * f(u[n]), Vector.zeros(dim))
            k2 = LinearSolver.solve_system(A, h * f(u[n] + k1), Vector.zeros(dim))
            u += [u[n] + b * (k1 + k2)]
            ODEIOSettings.out((n+1) / N)
        return u

    @staticmethod
    def implicit_3_order(f : VecFunc, u0 : Vector, h : float, N : int):
        u = [u0]
        g = 0.25
        J = Differetiator.J(f, u0)
        dim = u0.m
        bs = [0.115740740740741, 0.548927875243268, 0.335331647015991]
        a31 = 1.867943637803922
        a32 = 0.234444971391589
        A = Matrix.identity(dim) - h * g  * J
        for n in range(N):
            k1 = LinearSolver.solve_system(A, h * f(u[n]), Vector.zeros(dim))
            k2 = LinearSolver.solve_system(A, h * f(u[n] + 2*k1), Vector.zeros(dim))
            k3 = LinearSolver.solve_system(A, h * f(u[n] + a31*k1 + a32*k2), Vector.zeros(dim))
            u += [u[n] + bs[0]*k1 + bs[1]*k2 + bs[2]*k3]
            ODEIOSettings.out((n+1) / N)
        return u

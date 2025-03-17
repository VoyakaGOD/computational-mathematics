from typing import Callable
from matrixlib import *

VecTFunc = Callable[[float, Vector], Vector]
VecFunc = Callable[[Vector], Vector]

# solve Ax = f by Seidel, with initial approximation of x = x0
class LinearSolver:
    norm = get_Euclidian_norm
    eps = 1e-9

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
        return u

    @staticmethod
    def explicit_2_order(f : VecTFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h / 2, u[n] + h / 2 * k1)
            u += [u[n] + h * k2]
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
        return u

    @staticmethod
    def explicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [u[n] + h/12 * (23*f(t, u[n]) - 16*f(t-h, u[n-1]) + 5*f(t-2*h, u[n-2]))]
        return u

    @staticmethod
    def explicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [u[n] + h/24 * (55*f(t, u[n]) - 59*f(t-h, u[n-1]) + 37*f(t-2*h, u[n-2]) - 9*f(t-3*h, u[n-3]))]
        return u
    
    @staticmethod
    def implicit_1_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(N):
            t = t0 + n * h
            F = lambda x: x - h * f(t + h, x) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
        return u

    @staticmethod
    def implicit_2_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            F = lambda x: x - h/2 * (f(t + h, x) + f(t, u[n])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
        return u

    @staticmethod
    def implicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            F = lambda x: x - h/12 * (5*f(t + h, x) + 8*f(t, u[n]) - f(t-h, u[n-1])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
        return u
    
    @staticmethod
    def implicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            F = lambda x: x - h/24 * (9*f(t + h, x) + 19*f(t, u[n]) - 5*f(t-h, u[n-1]) + f(t-2*h, u[n-2])) - u[n]
            u += [NonlinearSolver.solve_system(F, u[n])]
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
        return u

    @staticmethod
    def explicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [-3/2*u[n] + 3*u[n-1] - 1/2*u[n-2] + 3*h * f(t, u[n])]
        return u

    @staticmethod
    def explicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [-10/3*u[n] + 6*u[n-1] - 2*u[n-2] + 1/3*u[n-3] + 4*h * f(t, u[n])]
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
        return u

    @staticmethod
    def implicit_3_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            F = lambda x: 11/6*x - 3*u[n] + 3/2*u[n-1] - 1/3*u[n-2] - h * f(t+h, x)
            u += [NonlinearSolver.solve_system(F, u[n])]
        return u

    @staticmethod
    def implicit_4_order(f : VecTFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            F = lambda x: 25/12*x - 4*u[n] + 3*u[n-1] - 4/3*u[n-2] + 1/4*u[n-3] - h * f(t+h, x)
            u += [NonlinearSolver.solve_system(F, u[n])]
        return u

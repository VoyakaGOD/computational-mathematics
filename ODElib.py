from typing import Callable
from matrixlib import *

VecFunc = Callable[[float, Vector], Vector]

# t_n = t_0 + nh, n in [0, N]
class RungeKuttaMethods:
    @staticmethod
    def explicit_1_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            u += [u[n] + h * f(t, u[n])]
        return u

    @staticmethod
    def explicit_2_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h / 2, u[n] + h / 2 * k1)
            u += [u[n] + h * k2]
        return u

    @staticmethod
    def explicit_3_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t + h/3, u[n] + h * k1/3)
            k3 = f(t + 2/3 * h, u[n] + 2/3 * h * k2)
            u += [u[n] + h * 0.25 * k1 + h * 0.75 * k3]
        return u

    @staticmethod
    def explicit_4_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
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
    def explicit_1_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        return RungeKuttaMethods.explicit_1_order(f, u0[0], h, N, t0)
    
    @staticmethod
    def explicit_2_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            u += [u[n] + h/2 * (3*f(t, u[n]) - f(t-h, u[n-1]))]
        return u

    @staticmethod
    def explicit_3_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [u[n] + h/12 * (23*f(t, u[n]) - 16*f(t-h, u[n-1]) + 5*f(t-2*h, u[n-2]))]
        return u

    @staticmethod
    def explicit_4_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [u[n] + h/24 * (55*f(t, u[n]) - 59*f(t-h, u[n-1]) + 37*f(t-2*h, u[n-2]) - 9*f(t-3*h, u[n-3]))]
        return u

# Backward Differentiation Formulas
# u0 = [u_0, ..., u_p], p - order of method
class BDF:
    def explicit_1_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        return AdamsMethods.explicit_1_order(f, u0, h, N, t0)
    
    def explicit_2_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(1, N):
            t = t0 + n * h
            u += [u[n-1] + 2*h * f(t, u[n])]
        return u

    def explicit_3_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(2, N):
            t = t0 + n * h
            u += [-3/2*u[n] + 3*u[n-1] - 1/2*u[n-2] + 3*h * f(t, u[n])]
        return u

    def explicit_4_order(f : VecFunc, u0 : list[Vector], h : float, N : int, t0 : float = 0):
        u = u0.copy()
        for n in range(3, N):
            t = t0 + n * h
            u += [145/47*u[n] - 114/47*u[n-1] + 5/47*u[n-2] + 1/47*u[n-3] + 13*h * f(t, u[n])]
        return u

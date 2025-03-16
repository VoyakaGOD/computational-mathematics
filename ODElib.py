from typing import Callable
from matrixlib import *

VecFunc = Callable[[float, Vector], Vector]

# t_n = t_0 + nh, n in [0, N]
class RungeKuttaMethods:
    def explicit_1_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            u += [u[n] + h * f(t, u[n])]
        return u
    
    def explicit_2_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t, u[n] + h / 2 * k1)
            u += [u[n] + h * k2]
        return u
    
    def explicit_3_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t, u[n] + h * k1/3)
            k3 = f(t, u[n] + 2/3 * h * k2)
            u += [u[n] + h * 0.25 * k1 + h * 0.75 * k3]
        return u

    def explicit_4_order(f : VecFunc, u0 : Vector, h : float, N : int, t0 : float = 0):
        u = [u0]
        for n in range(N):
            t = t0 + n * h
            k1 = f(t, u[n])
            k2 = f(t, u[n] + h * k1/2)
            k3 = f(t, u[n] + h * k2/2)
            k4 = f(t, u[n] + h * k3)
            u += [u[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]
        return u

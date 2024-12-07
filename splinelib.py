from math import sin, cos

def calculate_spline(x, next_x, prev_x, next_m, prev_m, alpha, beta, h):
    next_dx = next_x - x
    prev_dx = x - prev_x
    return (prev_m * next_dx**3 + next_m * prev_dx**3 + alpha * next_dx + beta * prev_dx) / h

def integrate_spline_with_oscillations(next_x, prev_x, next_m, prev_m, alpha, beta, h, k):
    c3 = next_m - prev_m
    c2 = 3 * (prev_m * next_x - next_m * prev_x)
    c1 = beta - alpha - 3 * (prev_m * next_x * next_x - next_m * prev_x * prev_x)
    c0 = alpha * next_x - beta * prev_x + prev_m * next_x * next_x * next_x - next_m * prev_x * prev_x * prev_x
    I0 = (cos(k * prev_x) - cos(k * next_x)) / k
    I1 = (sin(k*next_x) - k * next_x * cos(k * next_x) - sin(k*prev_x) + k * prev_x * cos(k * prev_x)) / k / k
    I2 = (2 * k * next_x) * sin(k * next_x) + (2 - k*k*next_x*next_x) * cos(k * next_x)
    I2 -= (2 * k * prev_x) * sin(k * prev_x) + (2 - k*k*prev_x*prev_x) * cos(k * prev_x)
    I2 /= k*k*k
    I3 = (3*k*k*next_x*next_x - 6) * sin(k * next_x) + (6 - k*k*next_x*next_x) * k * next_x * cos(k * next_x)
    I3 -= (3*k*k*prev_x*prev_x - 6) * sin(k * prev_x) + (6 - k*k*prev_x*prev_x) * k * prev_x * cos(k * prev_x)
    I3 /= k*k*k*k
    return (c0*I0 + c1*I1 + c2*I2 + c3*I3) / h

class NaturalSpline():
    def __init__(self, xs : list[float], ys : list[float], left_second_derivative : float = 0, right_second_derivative : float = 0):
        if len(xs) != len(ys):
            raise Exception(f"len(xs) = {len(xs)} != {len(ys)} = len(ys)")
        self.n = len(xs) - 1
        self.xs = xs
        self.h = [xs[i + 1] - xs[i] for i in range(self.n)]
        p = [0]
        r = [left_second_derivative / 6] # m[i] already divided by 6
        for i in range(1, self.n):
            Fi = (ys[i + 1] - ys[i]) / self.h[i] - (ys[i] - ys[i - 1]) / self.h[i-1]
            ai = self.h[i-1]
            bi = 2 * (self.h[i] + self.h[i-1])
            ci = self.h[i]
            denominator = bi - ai * p[-1]
            p += [ci / denominator]
            r += [(Fi - ai*r[-1]) / denominator]
        self.m = [0 for _ in range(self.n + 1)]
        self.m[self.n] = right_second_derivative / 6
        for i in range(self.n - 1, -1, -1):
            self.m[i] = r[i] - p[i] * self.m[i+1]
        self.alpha = [ys[i] - self.m[i]*self.h[i]*self.h[i] for i in range(self.n)]
        self.beta = [ys[i + 1] - self.m[i + 1]*self.h[i]*self.h[i] for i in range(self.n)]

    def __call__(self, x : float) -> float:
        for i in range(self.n):
            if self.xs[i] <= x <= self.xs[i+1]:
                return calculate_spline(x, self.xs[i+1], self.xs[i], self.m[i+1], self.m[i], self.alpha[i], self.beta[i], self.h[i])
        if x < self.xs[0]:
            return calculate_spline(x, self.xs[1], self.xs[0], self.m[1], self.m[0], self.alpha[0], self.beta[0], self.h[0])
        if x > self.xs[-1]:
            return calculate_spline(x, self.xs[-1], self.xs[-2], self.m[-1], self.m[-2], self.alpha[-1], self.beta[-1], self.h[-1])
        raise Exception("bad spline")

    #strictly integrate S(x)*sin(kx) from x_0 to x_n
    def integrate_with_oscillations(self, k : float) -> float:
        I = 0
        for i in range(self.n):
            I += integrate_spline_with_oscillations(self.xs[i+1], self.xs[i], self.m[i+1], self.m[i], 
                                                    self.alpha[i], self.beta[i], self.h[i], k)
        return I
    
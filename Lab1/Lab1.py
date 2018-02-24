import numpy as np

eps = 10 ** -8


def func(x):
    a, b, c, d, e, f = 1., 4., 0.001, 0., -1., 0.
    A = np.array([[a, c / 2],
                  [c / 2, b]])
    B = np.array([d, e])
    C = f
    return A.dot(x).dot(x) + B.dot(x) + C


def derivative(f, x, h=eps):
    grad = []
    for i in range(len(x)):
        y2, y1 = np.copy(x), np.copy(x)
        np.put(y1, [i], [y1[i] - h])
        np.put(y2, [i], [y2[i] + h])
        grad += [(f(y2) - f(y1)) / (2. * h)]
    return np.array(grad)


def minimize_one_dimension_golden(f, b=1 * 10 ** 5, a=-1 * 10 ** 5, eps=eps):
    F = (1. + 5 ** 0.5) / 2
    while abs(b - a) > eps:
        x1 = b - (b - a) / F
        x2 = a + (b - a) / F
        if f(x1) >= f(x2):
            a = x1
        else:
            b = x2
    return (a + b) / 2


def minimize_one_dimension(f, a0, method="golden"):
    if method == "golden":
        return minimize_one_dimension_golden(f)


def choose_step_fastest(f, x, h):
    def f_a(a, f=f, x=x, h=h):
        return f(x + a * h)

    return minimize_one_dimension(f_a, a0=0., method="golden")


def choose_step_drop(f, x, h, beta=1, λ=0.5):
    alpha = beta
    while f(x + alpha * h) > f(x):
        alpha *= λ
    return alpha


def choose_step(f, x, h, method="drop_step"):
    if method == "drop_step":
        return choose_step_drop(f, x, h)
    elif method == "fastest":
        return choose_step_fastest(f, x, h)


def norm(x):
    return sum([i ** 2 for i in x])**0.5


def minimize(f, x0, eps=eps):
    x = np.copy(x0)
    h = -derivative(f, x)
    alpha = choose_step(f, x, h)
    x1 = x + alpha * h
    while norm(x1 - x) > eps:
        x = np.copy(x1)
        h = -derivative(f, x)
        alpha = choose_step(f, x, h)
        x1 = x + alpha * h
    return x1


x0 = np.array([0., 0.])
print(minimize(func, x0))

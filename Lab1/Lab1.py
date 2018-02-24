import numpy as np

eps = 10 ** -7


def f(x):
    a, b, c, d, e, f = 1., 4., 0.001, 0., -1., 0.
    # A = np.array([[a, c / 2], [c / 2, d]])
    # B = np.array([d, e])
    # C = f
    # return A.dot(x).dot(x) + B.dot(x) + C
    return a * x[0] ** 2 + b * x[1] ** 2 + c * x[0] * x[1] + d * x[0] + e * x[1] + f


def derivative(f, x, h=eps):
    grad = [(f(np.array([x[0] + h, x[1]])) - f(np.array([x[0] - h, x[1]]))) / (2 * h),
            (f(np.array([x[0], x[1] + h])) - f(np.array([x[0], x[1] - h]))) / (2 * h)]
    # grad = []
    # for i in range(len(x)):
    #     y1 = np.copy(x)
    #     y2 = np.copy(x)
    #     grad += [(f(np.put(y2, [i], [y2[i] + h])) - f(np.put(y1, [i], [y1[i] - h]))) / (2 * h)]
    return np.array(grad)


def minimize_one_dimension_golden(f, b=1 * 10 ** 5, a=-1 * 10 ** 5, eps=eps):
    F = (1. + 5 ** 0.5) / 2
    while abs(b-a)>eps:
        x1 = b - (b - a) / F
        y1 = f(x1)
        x2 = a + (b - a) / F
        y2 = f(x2)
        if y1>=y2:
            a=x1
        else:
            b=x2
    return (a+b)/2


def minimize_one_dimension(f, a0, method="golden"):
    if method == "golden":
        return minimize_one_dimension_golden(f)
    # elif method == "fibonacci":
    #     return minimize_one_dimension_fibonacci(f, x0=a0)


def choose_step_fastest(f, x, h):
    def f_a(a, f=f, x=x, h=h):
        return f(x + a * h)

    return minimize_one_dimension(f_a, a0=0., method="golden")


def choose_step_drop(f, x, h, beta=1, λ =0.5):
    alpha = beta
    while f(x + alpha * h) > f(x):
        alpha *= λ
    return alpha


def choose_step(f, x, h, method="fastest"):
    if method == "drop_step":
        return choose_step_drop(f, x, h)
    elif method == "fastest":
        return choose_step_fastest(f, x, h)


def norm(x):
    sum = 0
    for i in x:
        sum += i ** 2
    return sum ** 0.5


def minimize(f, x0, eps=eps):
    x = np.copy(x0)
    h = -derivative(f, x)
    альфа = choose_step(f, x, h)
    x1 = x + альфа * h
    while norm(x1 - x) > eps:
        x = np.copy(x1)
        h = -derivative(f, x)
        x1 = x + choose_step(f, x, h) * h
    return x1


x0 = np.array([0., 0.])
print(minimize(f, x0))

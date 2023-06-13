import numpy as np

class BoundsException(Exception):
    pass

def velocity_position_model(m, g, theta, mu, beta, x):
    if mu > np.tan(theta) - 0.01:
        raise BoundsException()
    if beta == 0:
        return np.sqrt(2 * g * (np.sin(theta) - mu * np.cos(theta)) * x)
    else:
        return np.sqrt(m * g * (np.sin(theta) - mu * np.cos(theta)) * np.expm1(2 * beta * x / m)) / \
            (np.sqrt(beta) * np.exp(beta * x / m))


def acceleration_position_model(m, g, theta, mu, beta, x):
    return g * (np.sin(theta) - mu * np.cos(theta)) * np.exp(-2 * beta * x / m)


def ideal_velocity_position_model(g, theta, x):
    return np.sqrt(2 * g * np.sin(theta) * x)


def ideal_acceleration_position_model(g, theta, x):
    return np.repeat(g * np.sin(theta), len(x))

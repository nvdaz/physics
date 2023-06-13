import numpy as np

from functions import BoundsException, velocity_position_model, acceleration_position_model


def simultaneous_optimizer(params, x, vel, accel):
    mass = params["mass"]
    gravity = params["gravity"]
    angle = params["angle"]
    x0 = params["x0"]
    mu = params["mu"]
    beta = params["beta"]

    vel_fit, accel_fit = [], []
    try:
        vel_fit = velocity_position_model(mass, gravity, angle, mu, beta, np.maximum(x - x0, 0))
        accel_fit = acceleration_position_model(
            mass, gravity, angle, mu, beta, x - x0)
    except BoundsException as e:
        return np.concatenate((vel, accel))

    if np.isnan(vel_fit).any() or np.isnan(accel_fit).any():
        return np.concatenate((vel, accel))

    vel_residual = vel_fit - vel
    accel_residual = accel_fit - accel

    return np.concatenate(
        (vel_residual, accel_residual))

import sys
import os
import json
import numpy as np
from lmfit import create_params, fit_report, minimize
import matplotlib.pyplot as plt

from functions import velocity_position_model, acceleration_position_model, ideal_velocity_position_model, ideal_acceleration_position_model
from optimize import simultaneous_optimizer


dirname = sys.argv[1]

files = os.listdir(dirname)

files.remove("$calibration.csv")


mass, angle = 0, 0
gravity = 9.806

with open(f"{dirname}/$constants.json") as constants_file:
    constants = json.load(constants_file)

    mass = constants["mass"]
    angle = constants["angle"]

    assert mass > 0
    assert angle > 0


x0min, x0max = 0, 0

with open(f"{dirname}/$calibration.csv") as calibration_file:
    data = np.genfromtxt(calibration_file, dtype=float,
                         delimiter=",", skip_header=True)
    x = data[:, 1]

    assert len(x) > 0

    x0min = min(x)
    x0max = max(x)


x = np.array([])
v = np.array([])
a = np.array([])


for file in [file for file in files if file.endswith(".csv")]:
    with open(f"{dirname}/{file}") as data_file:
        data = np.genfromtxt(data_file, dtype=float,
                             delimiter=",", skip_header=True)
        x = np.append(x, data[:, 1])
        v = np.append(v, data[:, 2])
        a = np.append(a, data[:, 3])


params = create_params(mass={"value": mass, "vary": False},
                       gravity={"value": gravity, "vary": False},
                       angle={"value": angle, "vary": False},
                       x0={"value": 0, "min": x0min,
                           "max": x0max, "brute_step": 0.01},
                       mu={"value": 0, "vary": False, "min": 0,
                           "max": np.tan(angle) - 0.01, "brute_step": 0.01},
                       beta={"value": 0, "vary": True, "min": 0, "max": 2, "brute_step": 0.01})

result = minimize(simultaneous_optimizer, params,
                  args=(x, v, a), method="brute", nan_policy="omit", reduce_fcn="neglogcauchy")

result = minimize(simultaneous_optimizer, result.params,
                  args=(x, v, a), method="least_squares", nan_policy="omit", reduce_fcn="neglogcauchy")

print(fit_report(result))

params = result.params.valuesdict()

mass = params["mass"]
gravity = params["gravity"]
angle = params["angle"]
x0 = params["x0"]
mu = params["mu"]
beta = params["beta"]

with open(f"{dirname}/%fit.json", "w") as fit_file:
    json.dump(params, fit_file, indent=2)


include = [i > x0 for i in x]

fig, ax = plt.subplots(figsize=(16, 10))
twin = ax.twinx()

plot_x = np.linspace(x0, 2, 1000)

ax.scatter(x[include], v[include], c="lightskyblue", zorder=1)
twin.scatter(x[include], a[include], c="salmon", zorder=1)
ax.set_ylim([0, 1])
twin.set_ylim([0, 0.5])

plot_v_ideal = ideal_velocity_position_model(
    gravity, angle, plot_x - x0)
pvi, = ax.plot(plot_x, plot_v_ideal, label="Velocity (ideal)",
               c="powderblue", zorder=0)

plot_v = velocity_position_model(
    mass, gravity, angle, mu, beta, plot_x - x0)
pv, = ax.plot(plot_x, plot_v, label="Velocity", c="navy", zorder=2)

plot_a_ideal = ideal_acceleration_position_model(gravity, angle, plot_x - x0)
pai, = twin.plot(plot_x, plot_a_ideal,
                 label="Acceleration (ideal)", c="pink", zorder=0)

plot_a = acceleration_position_model(
    mass, gravity, angle, mu, beta, plot_x - x0)
pa, = twin.plot(plot_x, plot_a, label="Acceleration", c="crimson", zorder=2)


ax.set(xlabel="Position (m)", ylabel="Velocity (m/s)")
twin.set(ylabel="Acceleration (m/s²)")

ax.yaxis.label.set_color(pv.get_color())
twin.yaxis.label.set_color(pa.get_color())
ax.tick_params(axis="y", colors=pv.get_color())
twin.tick_params(axis="y", colors=pa.get_color())

plt.xlim([x0, 2])
plt.legend(handles=[pvi, pv, pai, pa], loc='best')
plt.savefig(f"{dirname}/%fit.png", dpi=300, bbox_inches="tight")
plt.show()

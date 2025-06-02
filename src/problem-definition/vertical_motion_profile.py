import numpy as np
import matplotlib.pyplot as plt

# Time vector setup for bang-bang motion (trapezoidal velocity profile)
n = 100
t_total = 6  # total time units
t = np.linspace(0, t_total, n * t_total)

# Define durations
T_a = 2  # time to accelerate
T_c = 2  # time at constant velocity
T_d = 2  # time to decelerate

# Initialize acceleration, velocity, and position arrays
acc = np.zeros_like(t)
vel = np.zeros_like(t)
pos = np.zeros_like(t)

# Set acceleration value (normalized)
a_max = 1

# Define acceleration profile
for i, ti in enumerate(t):
    if ti < T_a:
        acc[i] = a_max
    elif ti < T_a + T_c:
        acc[i] = 0
    else:
        acc[i] = -a_max

# Integrate acceleration to get velocity
vel = np.cumsum(acc) * (t[1] - t[0])
# Integrate velocity to get position
pos = np.cumsum(vel) * (t[1] - t[0])

# Markers for phase boundaries
phase_lines = [0, T_a, T_a + T_c, t_total]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

# Calculate maximum values for scaling
v_max = a_max * T_a  # maximum velocity
# Total displacement = area under velocity curve
# Area = (1/2 * T_a * v_max) + (T_c * v_max) + (1/2 * T_d * v_max)
s_max = 0.5 * T_a * v_max + T_c * v_max + 0.5 * T_d * v_max  # maximum displacement

axs[0].plot(t, pos, label="Position", color="blue")
axs[0].set_ylabel("$z_{ee}(t)$")
axs[0].set_yticks([0, s_max / 2, s_max])
axs[0].set_yticklabels(["$0$", "$\\frac{z_{ee,max}}{2}$", "$z_{ee,max}$"])
axs[0].grid(True)
# axs[0].set_title("n Profile")
for line in phase_lines:
    axs[0].axvline(x=line, color="gray", linestyle="--")
axs[0].legend()

axs[1].plot(t, vel, label="Velocity", color="green")
axs[1].set_ylabel("$\\dot{z}_{ee}(t)$")
axs[1].set_yticks([0, v_max / 2, v_max])
axs[1].set_yticklabels(["$0$", "$\\frac{\\dot{z}_{ee,max}}{2}$", "$\\dot{z}_{ee,max}$"])
axs[1].grid(True)
for line in phase_lines:
    axs[1].axvline(x=line, color="gray", linestyle="--")
axs[1].legend()

axs[2].plot(t, acc, label="Acceleration", color="red")
axs[2].set_xlabel("$t$")
axs[2].set_ylabel("$\\ddot{z}_{ee}(t)$")
axs[2].set_yticks([-a_max, 0, a_max])
axs[2].set_yticklabels(["$-\\ddot{z}_{ee,max}$", "$0$", "$\\ddot{z}_{ee,max}$"])
axs[2].grid(True)
for line in phase_lines:
    axs[2].axvline(x=line, color="gray", linestyle="--")
axs[2].legend()

# Set x-axis labels
axs[2].set_xticks(phase_lines)
axs[2].set_xticklabels(["$0$", "$T_a$", "$T_a + T_c$", "$T_{total}$"])

plt.tight_layout()
plt.show()

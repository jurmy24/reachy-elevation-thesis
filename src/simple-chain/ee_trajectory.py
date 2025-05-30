import numpy as np
import matplotlib.pyplot as plt

# Given values
z0 = 0.3541  # initial position (m)
z10 = 0.9701  # final position (m)
D = z10 - z0  # total displacement
v_max = 0.1  # max velocity (m/s)
a_max = 0.025  # max acceleration (m/s^2)
t_f = 10  # total time (s)

# Compute phase durations
T_a = v_max / a_max  # time to accelerate or decelerate
T_c = t_f - 2 * T_a  # time at constant velocity

# Time arrays for each phase
total_points = 300  # total number of points for the entire trajectory
points_per_second = total_points / t_f  # points per second

# Calculate number of points for each phase based on duration
n1 = int(T_a * points_per_second)  # acceleration phase
n2 = int(T_c * points_per_second)  # constant velocity phase
n3 = total_points - n1 - n2  # deceleration phase (remaining points)

t1 = np.linspace(0, T_a, n1)  # acceleration phase
t2 = np.linspace(T_a, T_a + T_c, n2)  # constant velocity phase
t3 = np.linspace(T_a + T_c, t_f, n3)  # deceleration phase

# Position, velocity, acceleration profiles
# Acceleration phase
z1 = z0 + 0.5 * a_max * t1**2
v1 = a_max * t1
a1 = np.full_like(t1, a_max)

# Constant velocity phase
z_Ta = z0 + 0.5 * a_max * T_a**2
z2 = z_Ta + v_max * (t2 - T_a)
v2 = np.full_like(t2, v_max)
a2 = np.zeros_like(t2)

# Deceleration phase
z_Tc = z_Ta + v_max * T_c
t3_shifted = t3 - (T_a + T_c)
z3 = z_Tc + v_max * t3_shifted - 0.5 * a_max * t3_shifted**2
v3 = v_max - a_max * t3_shifted
a3 = np.full_like(t3, -a_max)

# Combine all
t = np.concatenate([t1, t2, t3])
z = np.concatenate([z1, z2, z3])
v = np.concatenate([v1, v2, v3])
a = np.concatenate([a1, a2, a3])

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, z, label="Position z(t)", color="b")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, v, label="Velocity ẋ(t)", color="g")
plt.ylabel("Velocity (m/s)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, a, label="Acceleration ẍ(t)", color="r")
plt.ylabel("Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

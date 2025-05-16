import numpy as np
import matplotlib.pyplot as plt

# S-curve motion profile: jerk-limited, no cruise phase (triangular profile)

# Parameters
s_total = 0.6  # total distance (m)
j_max = 0.3  # max jerk (m/s^3)

# Compute Tj and total time
Tj = (3 * s_total / (2 * j_max)) ** (1 / 3)
T_total = 6 * Tj

# Time vector
dt = 0.001
t = np.arange(0, T_total, dt)

# Initialize profiles
jerk = np.zeros_like(t)
acc = np.zeros_like(t)
vel = np.zeros_like(t)
pos = np.zeros_like(t)

# Define segment times
T1 = Tj
T2 = T1 + Tj
T3 = T2 + Tj
T4 = T3 + Tj
T5 = T4 + Tj
T6 = T5 + Tj

# Build profiles
for i, ti in enumerate(t):
    if ti < T1:
        jerk[i] = j_max
    elif ti < T2:
        jerk[i] = 0
    elif ti < T3:
        jerk[i] = -j_max
    elif ti < T4:
        jerk[i] = -j_max
    elif ti < T5:
        jerk[i] = 0
    elif ti < T6:
        jerk[i] = j_max

# Integrate jerk to get acceleration, velocity, position
acc = np.cumsum(jerk) * dt
vel = np.cumsum(acc) * dt
pos = np.cumsum(vel) * dt

# Plot the profiles
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t, jerk, label="Jerk (m/s³)")
axs[0].set_ylabel("Jerk")
axs[0].legend()

axs[1].plot(t, acc, label="Acceleration (m/s²)", color="orange")
axs[1].set_ylabel("Acceleration")
axs[1].legend()

axs[2].plot(t, vel, label="Velocity (m/s)", color="green")
axs[2].set_ylabel("Velocity")
axs[2].legend()

axs[3].plot(t, pos, label="Position (m)", color="red")
axs[3].set_ylabel("Position")
axs[3].set_xlabel("Time (s)")
axs[3].legend()

plt.suptitle("Jerk-limited S-curve Motion Profile (no cruise)")
plt.tight_layout()
plt.show()

# Average velocity
# Using proper integration instead of just cumsum
for i in range(1, len(t)):
    acc[i] = acc[i - 1] + jerk[i] * dt
    vel[i] = vel[i - 1] + acc[i] * dt
    pos[i] = pos[i - 1] + vel[i] * dt

# Scale position to ensure it reaches s_total
scale_factor = s_total / pos[-1]
pos = pos * scale_factor
vel = vel * scale_factor
v_avg = s_total / T_total
print(f"Average velocity: {v_avg:.2f} m/s")
print(f"Average velocity from v's: {np.average(vel):.2f} m/s")

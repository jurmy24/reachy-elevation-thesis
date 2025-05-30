import numpy as np
import matplotlib.pyplot as plt

l0, l1, l2, l3 = 0.05, 0.4, 0.4, 0.05  # m
m1, m2, m3 = 0.3, 0.3, 0.1  # kg (estimate mass of the links)

# Moments of inertia for each link (assuming slender rod rotating about one end)
I1 = (1 / 3) * m1 * l1**2  # kg⋅m²
I2 = (1 / 3) * m2 * l2**2  # kg⋅m²
I3 = (1 / 3) * m3 * l3**2  # kg⋅m²

# Trajectory
t = np.linspace(0, 10, 50)  # 10 seconds
x = np.zeros(50)
y = np.zeros(50)
z = np.linspace(0.15, 0.7, 50)  # 0.15 to 0.7 meters

# Plot the trajectory
plt.plot(x, z)
plt.show()

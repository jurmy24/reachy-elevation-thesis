import numpy as np
import matplotlib.pyplot as plt


def compute_phase_durations(z0, z10, v_max, a_max, t_f):
    """Compute the durations of acceleration, constant velocity, and deceleration phases."""
    D = z10 - z0
    T_a = v_max / a_max  # time to accelerate or decelerate
    T_c = t_f - 2 * T_a  # time at constant velocity
    return T_a, T_c


def generate_trajectory(z0, z10, v_max, a_max, t_f, total_points=300):
    """Generate position, velocity, and acceleration trajectories."""
    T_a, T_c = compute_phase_durations(z0, z10, v_max, a_max, t_f)

    # Calculate number of points for each phase
    points_per_second = total_points / t_f
    n1 = int(T_a * points_per_second)  # acceleration phase
    n2 = int(T_c * points_per_second)  # constant velocity phase
    n3 = total_points - n1 - n2  # deceleration phase

    # Time arrays for each phase
    t1 = np.linspace(0, T_a, n1)
    t2 = np.linspace(T_a, T_a + T_c, n2)
    t3 = np.linspace(T_a + T_c, t_f, n3)

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

    return t, z, v, a


def plot_trajectory(t, z, v, a):
    """Plot position, velocity, and acceleration profiles."""
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


# Default trajectory parameters
DEFAULT_PARAMS = {
    "z0": 0.3541,  # initial position (m)
    "z10": 0.9701,  # final position (m)
    "v_max": 0.1,  # max velocity (m/s)
    "a_max": 0.025,  # max acceleration (m/s^2)
    "t_f": 10,  # total time (s)
}

# Generate default trajectory for direct import
t, z, v, a = generate_trajectory(**DEFAULT_PARAMS)

if __name__ == "__main__":
    # Generate and plot trajectory with default parameters
    t, z, v, a = generate_trajectory(**DEFAULT_PARAMS)
    plot_trajectory(t, z, v, a)

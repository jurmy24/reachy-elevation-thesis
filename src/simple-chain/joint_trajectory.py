import numpy as np
import matplotlib.pyplot as plt
from ik import numerical_ik
from jacobian import calculate_jacobian


def generate_bang_bang_trajectory(z0, zf, v_max, a_max, Tf, t):
    """
    Generate bang-bang trajectory for vertical motion.

    Args:
        z0: Initial position
        zf: Final position
        v_max: Maximum velocity
        a_max: Maximum acceleration
        Tf: Total time
        t: Time vector

    Returns:
        tuple: (z, zdot, zdotdot) position, velocity, acceleration arrays
    """
    Ta = v_max / a_max
    Tc = Tf - 2 * Ta

    z, zdot, zdotdot = [], [], []

    for ti in t:
        if ti < Ta:
            z.append(z0 + 0.5 * a_max * ti**2)
            zdot.append(a_max * ti)
            zdotdot.append(a_max)
        elif ti < Ta + Tc:
            z.append(z0 + 0.5 * a_max * Ta**2 + v_max * (ti - Ta))
            zdot.append(v_max)
            zdotdot.append(0)
        else:
            tau = ti - (Ta + Tc)
            z.append(
                z0
                + 0.5 * a_max * Ta**2
                + v_max * Tc
                + v_max * tau
                - 0.5 * a_max * tau**2
            )
            zdot.append(v_max - a_max * tau)
            zdotdot.append(-a_max)

    return np.array(z), np.array(zdot), np.array(zdotdot)


def compute_joint_trajectory(t, z_trajectory, phi_des, robot_params, initial_guess):
    """
    Compute joint angles using inverse kinematics for given trajectory.

    Args:
        t: Time vector
        z_trajectory: Desired z positions
        phi_des: Desired end-effector orientation
        robot_params: Tuple of (l0, l1, l2, l3) link lengths
        initial_guess: Initial guess for IK solver

    Returns:
        np.array: Joint angles for each time step
    """
    l0, l1, l2, l3 = robot_params
    theta_list = []
    current_guess = initial_guess.copy()

    for i in range(len(t)):
        x_des = 0
        z_des = z_trajectory[i]

        solution = numerical_ik(x_des, z_des, phi_des, l0, l1, l2, l3, current_guess)
        current_guess = solution
        theta_list.append(solution)

    return np.array(theta_list)


def compute_joint_velocities(theta_list, dt):
    """
    Compute joint velocities using finite difference.

    Args:
        theta_list: Joint angles array
        dt: Time step

    Returns:
        np.array: Joint velocities
    """
    return np.gradient(theta_list, dt, axis=0)


def compute_joint_accelerations(
    theta_list, theta_dot_list, z_dot, z_ddot, t, dt, robot_params
):
    """
    Compute joint accelerations using Jacobian method.

    Args:
        theta_list: Joint angles
        theta_dot_list: Joint velocities
        z_dot: Cartesian velocity in z
        z_ddot: Cartesian acceleration in z
        t: Time vector
        dt: Time step
        robot_params: Robot link lengths

    Returns:
        np.array: Joint accelerations
    """
    l0, l1, l2, l3 = robot_params
    theta_ddot_list = []

    for i in range(len(t)):
        J = calculate_jacobian(
            theta_list[i][0], theta_list[i][1], theta_list[i][2], l0, l1, l2, l3
        )

        # Compute Jacobian derivative
        J_dot = np.gradient(
            [
                calculate_jacobian(
                    theta_list[j][0], theta_list[j][1], theta_list[j][2], l0, l1, l2, l3
                )
                for j in range(max(0, i - 1), min(len(t), i + 2))
            ],
            dt,
            axis=0,
        )
        J_dot = J_dot[1] if len(J_dot) > 1 else np.zeros_like(J)

        # Cartesian velocities and accelerations
        dx = np.array([0, z_dot[i], 0])
        ddx = np.array([0, z_ddot[i], 0])

        # Solve for joint accelerations
        J_inv = np.linalg.pinv(J)
        dtheta = theta_dot_list[i]
        ddtheta = J_inv @ (ddx - J_dot @ dtheta)
        theta_ddot_list.append(ddtheta)

    return np.array(theta_ddot_list)


def analyze_trajectory(theta_dot_list, theta_ddot_list):
    """
    Analyze trajectory and print maximum velocities and accelerations.

    Args:
        theta_dot_list: Joint velocities
        theta_ddot_list: Joint accelerations
    """
    max_velocities = np.max(np.abs(theta_dot_list), axis=0)
    max_accelerations = np.max(np.abs(theta_ddot_list), axis=0)

    print("\nMaximum Joint Velocities (rad/s):")
    for i, v in enumerate(max_velocities):
        print(f"Joint {i+1}: {v:.4f}")

    print("\nMaximum Joint Accelerations (rad/s²):")
    for i, a in enumerate(max_accelerations):
        print(f"Joint {i+1}: {a:.4f}")


def plot_results(t, theta_list, theta_dot_list, theta_ddot_list):
    """
    Plot joint angles, velocities, and accelerations.

    Args:
        t: Time vector
        theta_list: Joint angles
        theta_dot_list: Joint velocities
        theta_ddot_list: Joint accelerations
    """
    plt.figure(figsize=(12, 8))

    # Joint angles
    plt.subplot(3, 1, 1)
    for i in range(3):
        plt.plot(t, theta_list[:, i], label=f"$\\theta_{i+1}$")
    plt.ylabel("Joint Angles (rad)")
    plt.grid()
    plt.legend()

    # Joint velocities
    plt.subplot(3, 1, 2)
    for i in range(3):
        plt.plot(t, theta_dot_list[:, i], label=f"$\\dot{{\\theta}}_{i+1}$")
    plt.ylabel("Joint Velocities (rad/s)")
    plt.grid()
    plt.legend()

    # Joint accelerations
    plt.subplot(3, 1, 3)
    for i in range(3):
        plt.plot(t, theta_ddot_list[:, i], label=f"$\\ddot{{\\theta}}_{i+1}$")
    plt.ylabel("Joint Accelerations (rad/s²)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Robot parameters
    robot_params = (0.2901, 0.4, 0.4, 0.05)  # l0, l1, l2, l3

    # Bang-bang profile parameters
    z0, zf = 0.3541, 0.9701
    v_max = 0.1
    a_max = 0.025
    Tf = 10

    # Time vector
    t = np.linspace(0, Tf, 500)
    dt = t[1] - t[0]

    # Generate bang-bang trajectory
    z, zdot, zdotdot = generate_bang_bang_trajectory(z0, zf, v_max, a_max, Tf, t)

    # Compute joint trajectory
    phi_des = np.pi / 2 - 0.0
    initial_guess = [0.0, 0.0, np.pi / 2 - 0.0]
    theta_list = compute_joint_trajectory(t, z, phi_des, robot_params, initial_guess)

    # Compute joint velocities
    theta_dot_list = compute_joint_velocities(theta_list, dt)

    # Compute joint accelerations
    theta_ddot_list = compute_joint_accelerations(
        theta_list, theta_dot_list, zdot, zdotdot, t, dt, robot_params
    )

    # Analyze trajectory
    analyze_trajectory(theta_dot_list, theta_ddot_list)

    # Plot results
    plot_results(t, theta_list, theta_dot_list, theta_ddot_list)


if __name__ == "__main__":
    main()

import numpy as np
from joint_trajectory import (
    compute_joint_accelerations,
    compute_joint_trajectory,
    compute_joint_velocities,
    generate_bang_bang_trajectory,
)


def calculate_inverse_dynamics(
    theta_list,
    theta_dot_list,
    theta_ddot_list,
    l0,
    l1,
    l2,
    l3,
    link_masses,
    link_inertias,
):
    """
    Calculate joint torques using Newton-Euler inverse dynamics

    Parameters:
    - theta_list: Joint angles [n_timesteps x 3]
    - theta_dot_list: Joint velocities [n_timesteps x 3]
    - theta_ddot_list: Joint accelerations [n_timesteps x 3]
    - l0, l1, l2, l3: Link lengths
    - link_masses: [m1, m2, m3] - masses of each link
    - link_inertias: [I1, I2, I3] - moments of inertia about CoM
    """

    n_timesteps = len(theta_list)
    joint_torques = np.zeros((n_timesteps, 3))

    g = np.array([0, 0, -9.81])  # Gravity vector

    for t in range(n_timesteps):
        theta = theta_list[t]
        theta_dot = theta_dot_list[t]
        theta_ddot = theta_ddot_list[t]

        # Calculate transformation matrices
        T_0_1 = get_transform_0_to_1(theta[0], l0, l1)
        T_0_2 = get_transform_0_to_2(theta[0], theta[1], l0, l1, l2)
        T_0_3 = get_transform_0_to_3(theta[0], theta[1], theta[2], l0, l1, l2, l3)

        # Initialize arrays for forward pass
        omega = [np.zeros(3) for _ in range(4)]  # Angular velocities
        alpha = [np.zeros(3) for _ in range(4)]  # Angular accelerations
        a_c = [np.zeros(3) for _ in range(4)]  # Linear accelerations of CoM

        # Forward pass: Calculate accelerations
        for i in range(1, 4):
            # Angular velocity (all rotations about y-axis)
            omega[i] = omega[i - 1] + np.array([0, theta_dot[i - 1], 0])

            # Angular acceleration
            alpha[i] = (
                alpha[i - 1]
                + np.array([0, theta_ddot[i - 1], 0])
                + np.cross(omega[i - 1], np.array([0, theta_dot[i - 1], 0]))
            )

            # Linear acceleration of center of mass
            # This requires calculating the position and Jacobian for each CoM
            a_c[i] = calculate_com_acceleration(
                i, theta, theta_dot, theta_ddot, l0, l1, l2, l3
            )

        # Initialize arrays for backward pass
        F = [np.zeros(3) for _ in range(4)]  # Forces at joints
        T = [np.zeros(3) for _ in range(4)]  # Torques at joints

        # Backward pass: Calculate forces and torques
        for i in range(3, 0, -1):
            # Force at center of mass (Newton's equation)
            F_c = link_masses[i - 1] * (a_c[i] - g)

            # Torque about center of mass (Euler's equation)
            T_c = link_inertias[i - 1] * alpha[i] + np.cross(
                omega[i], link_inertias[i - 1] * omega[i]
            )

            # Position vectors (from joint to CoM and to next joint)
            r_c = get_com_position_vector(i, theta, l1, l2, l3)
            r_next = get_next_joint_vector(i, l1, l2, l3) if i < 3 else np.zeros(3)

            # Force equilibrium
            F[i] = F_c + (F[i + 1] if i < 3 else np.zeros(3))

            # Torque equilibrium
            T[i] = (
                T_c
                + (T[i + 1] if i < 3 else np.zeros(3))
                + np.cross(r_c, F_c)
                + (np.cross(r_next, F[i + 1]) if i < 3 else np.zeros(3))
            )

        # Extract joint torques (only y-components matter for revolute joints)
        joint_torques[t] = [T[1][1], T[2][1], T[3][1]]

    return joint_torques


def calculate_com_acceleration(link_idx, theta, theta_dot, theta_ddot, l0, l1, l2, l3):
    """Calculate linear acceleration of center of mass for given link"""
    # This function needs to be implemented based on your specific
    # forward kinematics and Jacobian calculations
    # You'll use the Jacobian for the CoM position and its time derivative

    # Placeholder - replace with actual implementation
    J_com = calculate_com_jacobian(link_idx, theta, l0, l1, l2, l3)
    J_dot_com = calculate_com_jacobian_derivative(
        link_idx, theta, theta_dot, l0, l1, l2, l3
    )

    return J_com @ theta_ddot + J_dot_com @ theta_dot


def get_com_position_vector(link_idx, theta, l1, l2, l3):
    """Get position vector from joint to center of mass"""
    # For uniform links, CoM is typically at midpoint
    link_lengths = [l1, l2, l3]
    return np.array([link_lengths[link_idx - 1] / 2, 0, 0])


def get_next_joint_vector(link_idx, l1, l2, l3):
    """Get position vector from current joint to next joint"""
    link_lengths = [l1, l2, l3]
    return np.array([link_lengths[link_idx - 1], 0, 0])


# Helper functions for transformation matrices
def get_transform_0_to_1(theta1, l0, l1):
    # Implement based on your DH parameters
    pass


def get_transform_0_to_2(theta1, theta2, l0, l1, l2):
    # Implement based on your DH parameters
    pass


def get_transform_0_to_3(theta1, theta2, theta3, l0, l1, l2, l3):
    # Implement based on your DH parameters
    pass


# Example usage
if __name__ == "__main__":
    # Define link properties
    m1, m2, m3 = 1.0, 1.0, 40.77 + 0.5
    link_masses = [m1, m2, m3]  # kg - masses of links 1, 2, 3

    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05  # Link lengths (m)
    # l_ee_offset = 0.245 / 2  # End effector mass offset (m)

    # Moments of inertia (slender rods about one end)
    I1 = (1 / 3) * m1 * l1**2
    I2 = (1 / 3) * m2 * l2**2
    I3 = (1 / 3) * m3 * l3**2

    link_inertias = [I1, I2, I3]  # kg⋅m² - moments of inertia about CoM

    # Bang-bang profile parameters
    z0, zf = 0.3541, 0.9701
    v_max = 0.1
    a_max = 0.025
    Tf = 10

    # Time vector
    t = np.linspace(0, Tf, 500)
    dt = t[1] - t[0]

    phi_des = np.pi / 2 - 0.0
    initial_guess = [0.0, 0.0, np.pi / 2 - 0.0]
    robot_params = (l0, l1, l2, l3)

    # Generate bang-bang trajectory
    z, zdot, zdotdot = generate_bang_bang_trajectory(z0, zf, v_max, a_max, Tf, t)

    # Compute joint trajectory
    theta_list = compute_joint_trajectory(t, z, phi_des, robot_params, initial_guess)

    # Compute joint velocities
    theta_dot_list = compute_joint_velocities(theta_list, dt)

    # Compute joint accelerations
    theta_ddot_list = compute_joint_accelerations(
        theta_list, theta_dot_list, zdot, zdotdot, t, dt, robot_params
    )

    # Calculate torques
    torques = calculate_inverse_dynamics(
        theta_list,
        theta_dot_list,
        theta_ddot_list,
        l0,
        l1,
        l2,
        l3,
        link_masses,
        link_inertias,
    )

    print("Maximum joint torques:")
    print(f"Joint 1: {np.max(np.abs(torques[:, 0])):.3f} N⋅m")
    print(f"Joint 2: {np.max(np.abs(torques[:, 1])):.3f} N⋅m")
    print(f"Joint 3: {np.max(np.abs(torques[:, 2])):.3f} N⋅m")

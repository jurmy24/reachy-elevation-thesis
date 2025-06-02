import numpy as np
import math
import matplotlib.pyplot as plt
from ik import numerical_ik
from jacobian import calculate_jacobian
from ee_trajectory import t, z, v, a


def calculate_com_jacobian(theta1, theta2, theta3, l0, l1, l2, l3, link_num):
    """
    Calculate the Jacobian for the center of mass of a specific link.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    l0: base height offset
    l1, l2, l3: link lengths
    link_num: which link's COM jacobian to calculate (1, 2, or 3)

    Returns:
    J_com: 2x3 Jacobian matrix for the COM position (x, z only - planar motion)
    """

    if link_num == 1:
        # COM of link 1 is at l1/2 from joint 1
        com_dist = l1 / 2

        # Position of COM 1
        x_com = com_dist * math.cos(theta1)
        z_com = l0 + com_dist * math.sin(theta1)

        # Jacobian columns
        J1 = np.array(
            [
                -com_dist * math.sin(theta1),  # dx/dtheta1
                com_dist * math.cos(theta1),  # dz/dtheta1
            ]
        )
        J2 = np.array([0, 0])  # no dependency on theta2
        J3 = np.array([0, 0])  # no dependency on theta3

    elif link_num == 2:
        # COM of link 2 is at l2/2 from joint 2
        com_dist = l2 / 2

        # Position of joint 2
        x2 = l1 * math.cos(theta1)
        z2 = l0 + l1 * math.sin(theta1)

        # Position of COM 2
        x_com = x2 + com_dist * math.cos(theta1 + theta2)
        z_com = z2 + com_dist * math.sin(theta1 + theta2)

        # Jacobian columns
        J1 = np.array(
            [
                -l1 * math.sin(theta1) - com_dist * math.sin(theta1 + theta2),
                l1 * math.cos(theta1) + com_dist * math.cos(theta1 + theta2),
            ]
        )
        J2 = np.array(
            [
                -com_dist * math.sin(theta1 + theta2),
                com_dist * math.cos(theta1 + theta2),
            ]
        )
        J3 = np.array([0, 0])  # no dependency on theta3

    elif link_num == 3:
        # For link 3, we have:
        # - Link mass: 0.3 kg at l3/2 from joint 3
        # - Concentrated mass: 40.77 kg at l3 + 0.1225 m from joint 3
        # COM is the weighted average position

        m_link = 0.3  # kg
        m_load = 40.77  # kg
        m_total = m_link + m_load

        # Distances from joint 3
        d_link = l3 / 2  # center of link
        d_load = l3 + 0.1225  # concentrated load position

        # Weighted average COM distance from joint 3
        com_dist = (m_link * d_link + m_load * d_load) / m_total

        # Position of joint 3
        x3 = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
        z3 = l0 + l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)

        # Position of COM 3
        x_com = x3 + com_dist * math.cos(theta1 + theta2 + theta3)
        z_com = z3 + com_dist * math.sin(theta1 + theta2 + theta3)

        # Jacobian columns
        J1 = np.array(
            [
                -l1 * math.sin(theta1)
                - l2 * math.sin(theta1 + theta2)
                - com_dist * math.sin(theta1 + theta2 + theta3),
                l1 * math.cos(theta1)
                + l2 * math.cos(theta1 + theta2)
                + com_dist * math.cos(theta1 + theta2 + theta3),
            ]
        )
        J2 = np.array(
            [
                -l2 * math.sin(theta1 + theta2)
                - com_dist * math.sin(theta1 + theta2 + theta3),
                l2 * math.cos(theta1 + theta2)
                + com_dist * math.cos(theta1 + theta2 + theta3),
            ]
        )
        J3 = np.array(
            [
                -com_dist * math.sin(theta1 + theta2 + theta3),
                com_dist * math.cos(theta1 + theta2 + theta3),
            ]
        )

    # Combine into Jacobian matrix
    J_com = np.column_stack((J1, J2, J3))
    return J_com


def calculate_gravitational_torques(theta1, theta2, theta3, l0, l1, l2, l3, masses):
    """
    Calculate gravitational torques using Jacobian transpose method.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    l0: base height offset
    l1, l2, l3: link lengths
    masses: [m1, m2, m3] masses of the three links

    Returns:
    tau_g: 3x1 vector of gravitational torques
    """
    g = 9.81  # gravitational acceleration

    # Get Jacobians for each link's COM
    J_com1 = calculate_com_jacobian(theta1, theta2, theta3, l0, l1, l2, l3, 1)
    J_com2 = calculate_com_jacobian(theta1, theta2, theta3, l0, l1, l2, l3, 2)
    J_com3 = calculate_com_jacobian(theta1, theta2, theta3, l0, l1, l2, l3, 3)

    # Gravitational forces (only in z-direction)
    F_g1 = np.array([0, -masses[0] * g])
    F_g2 = np.array([0, -masses[1] * g])
    F_g3 = np.array([0, -masses[2] * g])

    # Calculate torques using Jacobian transpose
    tau_g1 = J_com1.T @ F_g1
    tau_g2 = J_com2.T @ F_g2
    tau_g3 = J_com3.T @ F_g3

    # Sum contributions from all links
    tau_g = tau_g1 + tau_g2 + tau_g3

    return tau_g


def calculate_inertial_torques(
    theta1, theta2, theta3, theta_dot, theta_ddot, l0, l1, l2, l3, masses, inertias
):
    """
    Calculate inertial torques including rotational and translational effects.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    theta_dot: joint velocities (rad/s)
    theta_ddot: joint accelerations (rad/s²)
    l0: base height offset
    l1, l2, l3: link lengths
    masses: [m1, m2, m3] masses of the three links
    inertias: [I1, I2, I3] rotational inertias about joint axes

    Returns:
    tau_in: 3x1 vector of inertial torques
    """
    tau_in = np.zeros(3)

    # Rotational inertia contribution
    tau_in += inertias * theta_ddot

    # Translational inertia contribution for each link's COM
    for i in range(3):
        link_num = i + 1

        # Get COM Jacobian
        J_com = calculate_com_jacobian(theta1, theta2, theta3, l0, l1, l2, l3, link_num)

        # Calculate COM acceleration
        com_accel = J_com @ theta_ddot

        # Calculate translational inertial forces
        F_inertial = masses[i] * com_accel

        # Convert to joint torques using Jacobian transpose
        tau_in += J_com.T @ F_inertial

    return tau_in


def analyze_torques():
    """
    Complete torque analysis for the trajectory.
    """
    # System parameters
    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05
    masses = np.array([1.0, 1.0, 40.77 + 0.3])  # kg

    # Calculate moments of inertia (assuming uniform rods)
    # I = (1/12) * m * l² for a rod about its end
    # But we want inertia about the joint axis, so we use parallel axis theorem
    # For a rod of length l and mass m, rotating about one end: I = (1/3) * m * l²
    inertias = np.array(
        [
            (1 / 3) * masses[0] * l1**2,
            (1 / 3) * masses[1] * l2**2,
            (1 / 3) * masses[2] * l3**2,
        ]
    )

    # Trajectory parameters
    pitch_rad = 0.0
    phi = np.pi / 2 - pitch_rad
    num_steps = len(t)

    # Arrays to store results
    theta_angles = np.zeros((num_steps, 3))
    theta_velocities = np.zeros((num_steps, 3))
    theta_accelerations = np.zeros((num_steps, 3))
    tau_gravitational = np.zeros((num_steps, 3))
    tau_inertial = np.zeros((num_steps, 3))
    tau_total = np.zeros((num_steps, 3))

    # Initial guess for IK
    initial_guess = np.array([0.0, 0.0, pitch_rad])

    print("Calculating torques for trajectory...")

    # Calculate joint angles for entire trajectory
    for i, zd in enumerate(z):
        try:
            solution = numerical_ik(0.0, zd, phi, l0, l1, l2, l3, initial_guess)
            theta_angles[i] = solution
            initial_guess = solution
        except:
            print(f"IK failed at step {i}, using previous solution")
            theta_angles[i] = theta_angles[i - 1] if i > 0 else initial_guess

    # Calculate velocities and accelerations using numerical differentiation
    dt = t[1] - t[0]

    # Joint velocities using central difference
    for j in range(3):
        theta_velocities[:, j] = np.gradient(theta_angles[:, j], dt)
        theta_accelerations[:, j] = np.gradient(theta_velocities[:, j], dt)

    # Calculate torques for each time step
    for i in range(num_steps):
        theta1, theta2, theta3 = theta_angles[i]
        theta_dot = theta_velocities[i]
        theta_ddot = theta_accelerations[i]

        # Gravitational torques
        tau_g = calculate_gravitational_torques(
            theta1, theta2, theta3, l0, l1, l2, l3, masses
        )
        tau_gravitational[i] = tau_g

        # Inertial torques
        tau_in = calculate_inertial_torques(
            theta1,
            theta2,
            theta3,
            theta_dot,
            theta_ddot,
            l0,
            l1,
            l2,
            l3,
            masses,
            inertias,
        )
        tau_inertial[i] = tau_in

        # Total torques
        tau_total[i] = tau_g + tau_in

    return (
        theta_angles,
        theta_velocities,
        theta_accelerations,
        tau_gravitational,
        tau_inertial,
        tau_total,
    )


def plot_torques(tau_gravitational, tau_inertial, tau_total):
    """
    Plot torque analysis results.
    """
    plt.figure(figsize=(10, 10))

    # Plot total torques
    plt.subplot(2, 2, 1)
    plt.plot(t, tau_total[:, 0], "r-", linewidth=2, label="Joint 1")
    plt.plot(t, tau_total[:, 1], "g-", linewidth=2, label="Joint 2")
    plt.plot(t, tau_total[:, 2], "b-", linewidth=2, label="Joint 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Torque (Nm)")
    plt.title("Total Joint Torques")
    plt.legend()
    plt.grid(True)

    # Plot gravitational torques
    plt.subplot(2, 2, 2)
    plt.plot(t, tau_gravitational[:, 0], "r--", linewidth=2, label="Joint 1")
    plt.plot(t, tau_gravitational[:, 1], "g--", linewidth=2, label="Joint 2")
    plt.plot(t, tau_gravitational[:, 2], "b--", linewidth=2, label="Joint 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Gravitational Torque (Nm)")
    plt.title("Gravitational Torques")
    plt.legend()
    plt.grid(True)

    # Plot inertial torques
    plt.subplot(2, 2, 3)
    plt.plot(t, tau_inertial[:, 0], "r:", linewidth=2, label="Joint 1")
    plt.plot(t, tau_inertial[:, 1], "g:", linewidth=2, label="Joint 2")
    plt.plot(t, tau_inertial[:, 2], "b:", linewidth=2, label="Joint 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Inertial Torque (Nm)")
    plt.title("Inertial Torques")
    plt.legend()
    plt.grid(True)

    # Plot comparison
    plt.subplot(2, 2, 4)
    plt.plot(t, np.abs(tau_total[:, 0]), "r-", linewidth=2, label="|τ₁|")
    plt.plot(t, np.abs(tau_total[:, 1]), "g-", linewidth=2, label="|τ₂|")
    plt.plot(t, np.abs(tau_total[:, 2]), "b-", linewidth=2, label="|τ₃|")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Torque (Nm)")
    plt.title("Absolute Joint Torques")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_torque_summary(tau_total):
    """
    Print summary statistics of torque analysis.
    """
    max_torques = np.max(np.abs(tau_total), axis=0)
    mean_torques = np.mean(np.abs(tau_total), axis=0)

    print("\n" + "=" * 50)
    print("TORQUE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Maximum absolute torques:")
    print(f"  Joint 1: {max_torques[0]:.2f} Nm")
    print(f"  Joint 2: {max_torques[1]:.2f} Nm")
    print(f"  Joint 3: {max_torques[2]:.2f} Nm")
    print()
    print(f"Mean absolute torques:")
    print(f"  Joint 1: {mean_torques[0]:.2f} Nm")
    print(f"  Joint 2: {mean_torques[1]:.2f} Nm")
    print(f"  Joint 3: {mean_torques[2]:.2f} Nm")
    print()
    print("LaTeX format for maximum torques:")
    print("\\[")
    print("|\\boldsymbol{\\tau}_{\\max}| =")
    print("\\begin{bmatrix}")
    print(f"{max_torques[0]:.2f} \\\\")
    print(f"{max_torques[1]:.2f} \\\\")
    print(f"{max_torques[2]:.2f}")
    print("\\end{bmatrix} \\, \\text{Nm}")
    print("\\]")


def plot_torques_for_report(tau_gravitational, tau_inertial, tau_total):
    """
    Create a publication-quality plot of joint torques for the report.
    """
    plt.figure(figsize=(11, 8))

    # Set font sizes for publication
    plt.rcParams.update({"font.size": 12})

    # Plot gravitational torques
    plt.subplot(3, 1, 1)
    plt.plot(t, tau_gravitational[:, 0], "r-", linewidth=2.5, label="Joint 1")
    plt.plot(t, tau_gravitational[:, 1], "g-", linewidth=2.5, label="Joint 2")
    plt.plot(t, tau_gravitational[:, 2], "b-", linewidth=2.5, label="Joint 3")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Gravitational (Nm)", fontsize=14)
    # plt.title('Gravitational Torques', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3, which="both", linestyle="-", linewidth=0.5)
    plt.minorticks_on()

    # Plot inertial torques
    plt.subplot(3, 1, 2)
    plt.plot(t, tau_inertial[:, 0], "r-", linewidth=2.5, label="Joint 1")
    plt.plot(t, tau_inertial[:, 1], "g-", linewidth=2.5, label="Joint 2")
    plt.plot(t, tau_inertial[:, 2], "b-", linewidth=2.5, label="Joint 3")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Inertial (Nm)", fontsize=14)
    # plt.title('Inertial Torques', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3, which="both", linestyle="-", linewidth=0.5)
    plt.minorticks_on()

    # Plot total torques
    plt.subplot(3, 1, 3)
    plt.plot(t, tau_total[:, 0], "r-", linewidth=2.5, label="Joint 1")
    plt.plot(t, tau_total[:, 1], "g-", linewidth=2.5, label="Joint 2")
    plt.plot(t, tau_total[:, 2], "b-", linewidth=2.5, label="Joint 3")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Total Torque (Nm)", fontsize=14)
    # plt.title('Total Joint Torques', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3, which="both", linestyle="-", linewidth=0.5)
    plt.minorticks_on()

    # Add some styling
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run complete torque analysis
    (
        theta_angles,
        theta_velocities,
        theta_accelerations,
        tau_gravitational,
        tau_inertial,
        tau_total,
    ) = analyze_torques()

    # Plot results for report
    plot_torques_for_report(tau_gravitational, tau_inertial, tau_total)

    # Plot detailed analysis
    plot_torques(tau_gravitational, tau_inertial, tau_total)

    # Print summary
    print_torque_summary(tau_total)

    # Additional analysis
    print("\n" + "=" * 50)
    print("ADDITIONAL ANALYSIS")
    print("=" * 50)

    # Power analysis
    power = np.sum(tau_total * theta_velocities, axis=1)
    max_power = np.max(np.abs(power))
    print(f"Maximum instantaneous power: {max_power:.2f} W")

    # RMS torques
    rms_torques = np.sqrt(np.mean(tau_total**2, axis=0))
    print(f"RMS torques:")
    print(f"  Joint 1: {rms_torques[0]:.2f} Nm")
    print(f"  Joint 2: {rms_torques[1]:.2f} Nm")
    print(f"  Joint 3: {rms_torques[2]:.2f} Nm")

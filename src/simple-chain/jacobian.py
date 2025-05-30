import numpy as np
import math
import matplotlib.pyplot as plt
from ik import numerical_ik
from ee_trajectory import t, v, z  # Import trajectory data


# This it eh properly setup jacobian matrix
def calculate_jacobian(theta1, theta2, theta3, l0, l1, l2, l3):
    """
    Calculate the geometric Jacobian for the 3-DOF planar manipulator.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    l0: base height offset
    l1, l2, l3: link lengths

    Returns:
    J: 3x3 Jacobian matrix (2 for position, 1 for orientation)
    x, z, phi
    """
    # Calculate positions of each joint
    x1 = l1 * math.cos(theta1)
    z1 = l0 + l1 * math.sin(theta1)

    x2 = x1 + l2 * math.cos(theta1 + theta2)
    z2 = z1 + l2 * math.sin(theta1 + theta2)

    x3 = x2 + l3 * math.cos(theta1 + theta2 + theta3)
    z3 = z2 + l3 * math.sin(theta1 + theta2 + theta3)

    # End-effector position
    xe = x3
    ze = z3

    # NOTE: the final row is just 1,1,1 because phi = theta_1 + theta_2 + theta_3

    # Calculate Jacobian columns
    # For planar manipulator, each column is [dz/dθi, -dx/dθi, 1]^T
    J1 = np.array(
        [
            -l1 * math.sin(theta1)
            - l2 * math.sin(theta1 + theta2)
            - l3 * math.sin(theta1 + theta2 + theta3),
            l1 * math.cos(theta1)
            + l2 * math.cos(theta1 + theta2)
            + l3 * math.cos(theta1 + theta2 + theta3),
            1,
        ]
    )

    J2 = np.array(
        [
            -l2 * math.sin(theta1 + theta2) - l3 * math.sin(theta1 + theta2 + theta3),
            l2 * math.cos(theta1 + theta2) + l3 * math.cos(theta1 + theta2 + theta3),
            1,
        ]
    )

    J3 = np.array(
        [
            -l3 * math.sin(theta1 + theta2 + theta3),
            l3 * math.cos(theta1 + theta2 + theta3),
            1,
        ]
    )

    # Combine into Jacobian matrix
    J = np.column_stack((J1, J2, J3))

    return J


def calculate_joint_velocities(J, end_effector_velocity):
    """
    Calculate joint velocities from end-effector velocity using the Jacobian.

    Parameters:
    J: 3x3 Jacobian matrix
    end_effector_velocity: 3x1 vector [vx, vz, ω]^T

    Returns:
    joint_velocities: 3x1 vector of joint velocities
    """
    # Check if Jacobian is singular
    if abs(np.linalg.det(J)) < 1e-6:
        raise ValueError(
            "Jacobian is singular - manipulator is in a singular configuration"
        )

    # Calculate joint velocities using inverse of Jacobian
    joint_velocities = np.linalg.solve(J, end_effector_velocity)
    return joint_velocities


def calculate_end_effector_velocity(J, joint_velocities):
    """
    Calculate end-effector velocity from joint velocities using the Jacobian.

    Parameters:
    J: 3x3 Jacobian matrix
    joint_velocities: 3x1 vector of joint velocities

    Returns:
    end_effector_velocity: 3x1 vector [vx, vz, ω]^T
    """
    return J @ joint_velocities


# Example usage
if __name__ == "__main__":
    # Link lengths
    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05

    # trajectory
    pitch_rad = 0.0
    phi = np.pi / 2 - pitch_rad  # pi/2 is vertical, positive pitch tilts forward
    num_steps = len(t)  # Use same number of steps as trajectory
    z_values = z  # Use z values from trajectory
    time = t  # Use time values from trajectory

    # Use previous solution as initial guess for next step
    initial_guess = np.array([0.0, 0.0, pitch_rad])  # Start with a reasonable guess

    # Arrays to store joint velocities
    theta1_velocities = []
    theta2_velocities = []
    theta3_velocities = []

    for i, zd in enumerate(z_values):
        solution = numerical_ik(0.0, zd, phi, l0, l1, l2, l3, initial_guess)
        theta1, theta2, theta3 = solution
        initial_guess = solution  # Update initial guess for next iteration

        # Calculate Jacobian
        J = calculate_jacobian(theta1, theta2, theta3, l0, l1, l2, l3)

        # Calculate joint velocities using actual velocity from trajectory
        end_effector_velocity = np.array(
            [0.0, v[i], 0.0]
        )  # Use actual velocity from trajectory
        joint_velocities = calculate_joint_velocities(J, end_effector_velocity)

        # Store joint velocities
        theta1_velocities.append(joint_velocities[0])
        theta2_velocities.append(joint_velocities[1])
        theta3_velocities.append(joint_velocities[2])

    # Plot joint velocities over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, theta1_velocities, label="θ₁ velocity")
    plt.plot(time, theta2_velocities, label="θ₂ velocity")
    plt.plot(time, theta3_velocities, label="θ₃ velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angular Velocity (rad/s)")
    plt.title("Joint Angular Velocities During Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate accelerations using central difference method
    dt = time[1] - time[0]  # Time step
    theta1_acc = np.gradient(theta1_velocities, dt)
    theta2_acc = np.gradient(theta2_velocities, dt)
    theta3_acc = np.gradient(theta3_velocities, dt)

    # Plot accelerations
    plt.figure(figsize=(10, 6))
    plt.plot(time, theta1_acc, label="θ₁ acceleration")
    plt.plot(time, theta2_acc, label="θ₂ acceleration")
    plt.plot(time, theta3_acc, label="θ₃ acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angular Acceleration (rad/s²)")
    plt.title("Joint Angular Accelerations During Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print maximum absolute angular velocities
    max_theta1 = max(abs(np.array(theta1_velocities)))
    max_theta2 = max(abs(np.array(theta2_velocities)))
    max_theta3 = max(abs(np.array(theta3_velocities)))

    print("\nMaximum absolute angular velocities:")
    print(f"θ₁: {max_theta1:.3f} rad/s")
    print(f"θ₂: {max_theta2:.3f} rad/s")
    print(f"θ₃: {max_theta3:.3f} rad/s")

    # Calculate and print maximum absolute angular accelerations
    max_theta1_acc = max(abs(theta1_acc))
    max_theta2_acc = max(abs(theta2_acc))
    max_theta3_acc = max(abs(theta3_acc))

    print("\nMaximum absolute angular accelerations:")
    print(f"θ₁: {max_theta1_acc:.3f} rad/s²")
    print(f"θ₂: {max_theta2_acc:.3f} rad/s²")
    print(f"θ₃: {max_theta3_acc:.3f} rad/s²")

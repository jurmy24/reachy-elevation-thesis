import numpy as np
import math
from fk import forward_kinematics, extract_position_orientation


def calculate_jacobian(theta1, theta2, theta3, l0, l1, l2, l3):
    """
    Calculate the geometric Jacobian for the 3-DOF planar manipulator.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    l0: base height offset
    l1, l2, l3: link lengths

    Returns:
    J: 3x3 Jacobian matrix (2 for position, 1 for orientation)
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


def calculate_joint_accelerations(
    J, J_dot, end_effector_velocity, end_effector_acceleration
):
    """
    Calculate joint accelerations from end-effector acceleration.

    Parameters:
    J: 3x3 Jacobian matrix
    J_dot: 3x3 time derivative of Jacobian matrix
    end_effector_velocity: 3x1 vector [vx, vz, ω]^T
    end_effector_acceleration: 3x1 vector [ax, az, α]^T

    Returns:
    joint_accelerations: 3x1 vector of joint accelerations
    """
    # Check if Jacobian is singular
    if abs(np.linalg.det(J)) < 1e-6:
        raise ValueError(
            "Jacobian is singular - manipulator is in a singular configuration"
        )

    # Calculate joint velocities first
    joint_velocities = np.linalg.solve(J, end_effector_velocity)

    # Calculate joint accelerations
    # J * q_ddot = x_ddot - J_dot * q_dot
    joint_accelerations = np.linalg.solve(
        J, end_effector_acceleration - J_dot @ joint_velocities
    )

    return joint_accelerations


# Example usage
if __name__ == "__main__":
    # Test parameters
    theta1 = math.pi / 4
    theta2 = math.pi / 6
    theta3 = -math.pi / 3
    l0, l1, l2, l3 = 0.05, 0.4, 0.4, 0.05

    # Calculate Jacobian
    J = calculate_jacobian(theta1, theta2, theta3, l0, l1, l2, l3)
    print("Jacobian matrix:")
    print(J)

    # Test velocity calculations
    end_effector_velocity = np.array([0.1, 0.1, 0.1])  # Example end-effector velocity
    joint_velocities = calculate_joint_velocities(J, end_effector_velocity)
    print("\nJoint velocities:")
    print(joint_velocities)

    # Verify by calculating back to end-effector velocity
    calculated_velocity = calculate_end_effector_velocity(J, joint_velocities)
    print("\nCalculated end-effector velocity:")
    print(calculated_velocity)
    print("\nError in velocity calculation:")
    print(end_effector_velocity - calculated_velocity)

import numpy as np
import math


def dh_transform(theta, d, r, alpha):
    """
    Calculate the transformation matrix from DH parameters.

    Parameters:
    theta: joint angle (radians)
    d: link offset along z-axis
    r: link length along x-axis
    alpha: link twist around x-axis (radians)

    Returns:
    4x4 transformation matrix
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    T = np.array(
        [
            [ct, -st * ca, st * sa, r * ct],
            [st, ct * ca, -ct * sa, r * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )

    return T


# Alternative direct calculation (for verification)
def forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3):
    """
    Direct calculation of forward kinematics for planar manipulator in xz plane.
    This is equivalent to the DH method but more explicit for planar case.
    """
    # End-effector position in xz plane
    x = (
        l1 * math.cos(theta1)
        + l2 * math.cos(theta1 + theta2)
        + l3 * math.cos(theta1 + theta2 + theta3)
    )
    z = l0 + (
        l1 * math.sin(theta1)
        + l2 * math.sin(theta1 + theta2)
        + l3 * math.sin(theta1 + theta2 + theta3)
    )

    # End-effector orientation
    phi = theta1 + theta2 + theta3

    return np.array([x, 0, z]), phi


def forward_kinematics(theta1, theta2, theta3, l0, l1, l2, l3):
    """
    Calculate forward kinematics for 3-DOF planar manipulator in xz plane.

    Parameters:
    theta1, theta2, theta3: joint angles (radians)
    l0: base height offset
    l1, l2, l3: link lengths

    Returns:
    T_total: 4x4 transformation matrix from base to end-effector
    T_matrices: list of individual transformation matrices
    """

    # Base transformation matrix (translates by l0 in z)
    T_base = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, l0], [0, 0, 0, 1]])

    # For planar motion in xz plane, we need to rotate the coordinate frame
    # to align with the xz plane. We'll use a 90-degree rotation around x-axis
    R_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    T_rotate = np.eye(4)
    T_rotate[0:3, 0:3] = R_x

    # DH parameters for each joint (now in xz plane)
    # Joint 1: theta1, d=0, r=l1, alpha=0
    T_0_1 = dh_transform(theta1, 0, l1, 0)

    # Joint 2: theta2, d=0, r=l2, alpha=0
    T_1_2 = dh_transform(theta2, 0, l2, 0)

    # Joint 3: theta3, d=0, r=l3, alpha=0
    T_2_3 = dh_transform(theta3, 0, l3, 0)

    # Calculate cumulative transformations
    T_0_1_total = T_base @ T_rotate @ T_0_1
    T_0_2_total = T_0_1_total @ T_1_2
    T_0_3_total = T_0_2_total @ T_2_3

    # Store all matrices for debugging/visualization
    T_matrices = {
        "T_base": T_base,
        "T_rotate": T_rotate,
        "T_0_1": T_0_1,
        "T_1_2": T_1_2,
        "T_2_3": T_2_3,
        "T_0_1_total": T_0_1_total,
        "T_0_2_total": T_0_2_total,
        "T_0_3_total": T_0_3_total,
    }

    return T_0_3_total, T_matrices


def extract_position_orientation(T):
    """
    Extract position and orientation from transformation matrix.
    """
    position = T[0:3, 3]
    rotation_matrix = T[0:3, 0:3]

    return position, rotation_matrix


# Verification function
def verify_calculations():
    """Verify DH method against direct calculation."""
    theta1, theta2, theta3 = math.pi / 4, math.pi / 6, -math.pi / 3
    l0, l1, l2, l3 = 0.4, 1.0, 0.8, 0.6

    # DH method
    T_final, _ = forward_kinematics(theta1, theta2, theta3, l0, l1, l2, l3)
    pos_dh, rot_dh = extract_position_orientation(T_final)

    # Direct method
    pos_direct, phi_direct = forward_kinematics_direct(
        theta1, theta2, theta3, l0, l1, l2, l3
    )

    print("=== Verification ===")
    print("DH method position:", pos_dh)
    print("Direct method position:", pos_direct)
    print("Position difference:", np.linalg.norm(pos_dh - pos_direct))
    print(f"DH orientation angle: {math.atan2(rot_dh[1,0], rot_dh[0,0]):.4f} rad")
    print(f"Direct orientation angle: {phi_direct:.4f} rad")


# Example usage
if __name__ == "__main__":
    # Define joint angles (in radians)
    theta1 = math.pi / 4  # 45 degrees
    theta2 = math.pi / 6  # 30 degrees
    theta3 = -math.pi / 3  # -60 degrees

    # Define link lengths
    l0 = 0.4
    l1 = 1.0
    l2 = 0.8
    l3 = 0.6

    # Calculate forward kinematics
    T_final, T_matrices = forward_kinematics(theta1, theta2, theta3, l0, l1, l2, l3)

    # Extract end-effector position and orientation
    position, rotation = extract_position_orientation(T_final)

    # Print results
    print("=== Forward Kinematics Results ===")
    print(
        f"Joint angles: θ1={math.degrees(theta1):.1f}°, θ2={math.degrees(theta2):.1f}°, θ3={math.degrees(theta3):.1f}°"
    )
    print(f"Link lengths: l1={l1}, l2={l2}, l3={l3}")
    print()

    print("Final transformation matrix (T_0_3):")
    print(T_final)
    print()

    print("End-effector position:")
    print(f"x = {position[0]:.4f}")
    print(f"y = {position[1]:.4f}")
    print(f"z = {position[2]:.4f}")
    print()

    print("End-effector orientation (rotation matrix):")
    print(rotation)
    print()

    # Print individual transformation matrices for debugging
    print("=== Individual Transformation Matrices ===")
    for name, matrix in T_matrices.items():
        print(f"{name}:")
        print(matrix)
        print()

    verify_calculations()

import numpy as np
import matplotlib.pyplot as plt
from jacobian import calculate_jacobian


def analyze_singularities(l0, l1, l2, l3):
    """
    Analyze singularities of the 3-DOF planar manipulator by calculating
    the Jacobian determinant across different joint configurations.

    Parameters:
    l0, l1, l2, l3: link lengths
    """
    # Create grid of joint angles
    theta1_range = np.linspace(-np.pi, np.pi, 100)
    theta2_range = np.linspace(-np.pi, np.pi, 100)
    theta3_range = np.linspace(-np.pi, np.pi, 100)

    # Initialize arrays to store results
    det_values = []
    singular_configs = []

    # Calculate determinant for each configuration
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            for theta3 in theta3_range:
                J = calculate_jacobian(theta1, theta2, theta3, l0, l1, l2, l3)
                det = np.linalg.det(J)
                det_values.append(det)

                # Store configurations close to singularity
                if abs(det) < 1e-3:
                    singular_configs.append((theta1, theta2, theta3, det))

    # Plot histogram of determinant values
    plt.figure(figsize=(10, 6))
    plt.hist(det_values, bins=50)
    plt.xlabel("Jacobian Determinant")
    plt.ylabel("Frequency")
    plt.title("Distribution of Jacobian Determinant Values")
    plt.grid(True)
    plt.show()

    # Print singular configurations
    print("\nSingular Configurations Found:")
    print("theta1 (rad) | theta2 (rad) | theta3 (rad) | determinant")
    print("-" * 60)
    for config in singular_configs:
        print(
            f"{config[0]:11.3f} | {config[1]:11.3f} | {config[2]:11.3f} | {config[3]:11.3e}"
        )

    # Analyze specific cases
    print("\nAnalyzing specific configurations:")

    # Case 1: All joints aligned (theta2 = 0)
    J_aligned = calculate_jacobian(0, 0, 0, l0, l1, l2, l3)
    det_aligned = np.linalg.det(J_aligned)
    print(f"\nAll joints aligned (0,0,0):")
    print(f"Determinant: {det_aligned:.3e}")

    # Case 2: Joints in a straight line (theta2 = pi)
    J_straight = calculate_jacobian(0, np.pi, 0, l0, l1, l2, l3)
    det_straight = np.linalg.det(J_straight)
    print(f"\nJoints in straight line (0,π,0):")
    print(f"Determinant: {det_straight:.3e}")


if __name__ == "__main__":
    # Link lengths (same as in jacobian.py)
    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05

    # Analyze singularities
    analyze_singularities(l0, l1, l2, l3)

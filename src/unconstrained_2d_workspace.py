import numpy as np
import matplotlib.pyplot as plt
from fk import forward_kinematics, extract_position_orientation


def draw_links(theta1, theta2, theta3, l0, l1, l2, l3, ax):
    """
    Draw the links of the manipulator for a given configuration.

    Parameters:
    theta1, theta2, theta3: joint angles
    l0, l1, l2, l3: link lengths
    ax: matplotlib axis to draw on
    """
    # Draw the base link (l0)
    ax.plot([0, 0], [0, l0], "k-", linewidth=3, label="Base Link")

    # Calculate positions of all joints
    T1, _ = forward_kinematics(theta1, 0, 0, l0, l1, 0, 0)
    T2, _ = forward_kinematics(theta1, theta2, 0, l0, l1, l2, 0)
    T3, orientation = forward_kinematics(theta1, theta2, theta3, l0, l1, l2, l3)

    pos1, _ = extract_position_orientation(T1)
    pos2, _ = extract_position_orientation(T2)
    pos3, _ = extract_position_orientation(T3)

    # Draw links
    ax.plot([0, pos1[0]], [l0, pos1[2]], "b-", linewidth=3, label="Link 1")
    ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], "g-", linewidth=3, label="Link 2")
    ax.plot([pos2[0], pos3[0]], [pos2[2], pos3[2]], "r-", linewidth=3, label="Link 3")

    # Draw joints
    ax.plot(0, 0, "ko", markersize=8)  # Origin
    ax.plot(0, l0, "ko", markersize=8)  # Base joint
    ax.plot(pos1[0], pos1[2], "ko", markersize=8)  # Joint 1
    ax.plot(pos2[0], pos2[2], "ko", markersize=8)  # Joint 2
    # ax.plot(pos3[0], pos3[2], "ko", markersize=8)  # End-effector


def plot_workspace(l0, l1, l2, l3, num_points=10000):
    """
    Plot the 2D workspace of the manipulator in the xz plane.

    Parameters:
    l0, l1, l2, l3: link lengths
    num_points: number of random configurations to sample
    """
    # Generate random joint angles
    theta1_samples = np.random.uniform(-np.pi, np.pi, num_points)
    theta2_samples = np.random.uniform(-np.pi, np.pi, num_points)
    theta3_samples = np.random.uniform(-np.pi, np.pi, num_points)

    # Store end-effector positions
    x_positions = []
    z_positions = []

    # Calculate end-effector positions for each configuration
    for theta1, theta2, theta3 in zip(theta1_samples, theta2_samples, theta3_samples):
        T_final, _ = forward_kinematics(theta1, theta2, theta3, l0, l1, l2, l3)
        position, _ = extract_position_orientation(T_final)
        x_positions.append(position[0])  # x coordinate
        z_positions.append(position[2])  # z coordinate

    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.scatter(x_positions, z_positions, s=1, alpha=0.5, label="Workspace points")

    # Plot the base position
    plt.plot(0, l0, "ro", markersize=10, label="Base")

    # Draw the links for a specific configuration (using the first sample)
    draw_links(
        theta1_samples[0], theta2_samples[0], theta3_samples[0], l0, l1, l2, l3, ax
    )

    # Add labels and title
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("2D Workspace of 3-DOF Planar Manipulator")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    # Add link lengths to the plot
    plt.text(
        0.02,
        0.98,
        f"Link lengths:\nl0 = {l0:.2f}m\nl1 = {l1:.2f}m\nl2 = {l2:.2f}m\nl3 = {l3:.2f}m",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return plt


if __name__ == "__main__":
    # Define link lengths (same as in fk.py example)
    l0 = 0.05  # base height
    l1 = 0.4  # first link
    l2 = 0.4  # second link
    l3 = 0.05  # third link

    # Generate and show the workspace plot
    plt = plot_workspace(l0, l1, l2, l3)
    plt.show()

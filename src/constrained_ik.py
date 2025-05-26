import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from ik import forward_kinematics_direct


def constrained_ik(xd, zd, phi, l0, l1, l2, l3, initial_guess):
    """
    Solve IK with constraints that the mechanism can only move vertically
    with a maximum pitch of 30 degrees in either direction.

    Note: phi represents the end-effector orientation where:
    - phi = pi/2 (90°) means pointing vertically upward (pitch = 0°)
    - phi = pi/2 + pitch means tilted forward by 'pitch' angle
    - phi = pi/2 - pitch means tilted backward by 'pitch' angle
    """

    def objective(thetas):
        theta1, theta2, theta3 = thetas
        xz_phi = forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3)
        return np.sum((xz_phi - np.array([xd, zd, phi])) ** 2)

    def vertical_constraint(thetas):
        theta1, theta2, theta3 = thetas
        xz_phi = forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3)
        # Constrain x to be close to 0 (vertical motion)
        return 0.05 - abs(xz_phi[0])  # Allow small x deviation

    def pitch_constraint(thetas):
        theta1, theta2, theta3 = thetas
        xz_phi = forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3)
        # The end-effector orientation phi should be within ±30° from vertical (pi/2)
        deviation_from_vertical = abs(xz_phi[2] - np.pi / 2)
        return np.pi / 6 - deviation_from_vertical  # Allow max 30° deviation

    # Set joint limits in radians
    bounds = [(-np.pi, np.pi) for _ in range(3)]

    # Define constraints
    constraints = [
        {"type": "ineq", "fun": vertical_constraint},  # x should be close to 0
        {
            "type": "ineq",
            "fun": pitch_constraint,
        },  # pitch within ±30 degrees from vertical
    ]

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError("Constrained IK did not converge.")
    return result.x


def plot_mechanism(theta1, theta2, theta3, l0, l1, l2, l3, ax=None):
    """Plot the 3-link mechanism in its current configuration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Calculate joint positions
    x0, z0 = 0, l0
    x1 = x0 + l1 * np.cos(theta1)
    z1 = z0 + l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    z2 = z1 + l2 * np.sin(theta1 + theta2)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    z3 = z2 + l3 * np.sin(theta1 + theta2 + theta3)

    # Plot links
    ax.plot([x0, x1], [z0, z1], "b-", linewidth=2)
    ax.plot([x1, x2], [z1, z2], "g-", linewidth=2)
    ax.plot([x2, x3], [z2, z3], "r-", linewidth=2)

    # Plot joints
    ax.plot([x0, x1, x2, x3], [z0, z1, z2, z3], "ko")

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("3-Link Mechanism Configuration")


def plot_mechanism_transparent(
    theta1, theta2, theta3, l0, l1, l2, l3, ax, alpha=0.3, color_offset=0
):
    """Plot the 3-link mechanism with transparency."""

    # Calculate joint positions
    x0, z0 = 0, l0
    x1 = x0 + l1 * np.cos(theta1)
    z1 = z0 + l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    z2 = z1 + l2 * np.sin(theta1 + theta2)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    z3 = z2 + l3 * np.sin(theta1 + theta2 + theta3)

    # Define colors for different pitch angles
    colors = ["blue", "green", "red"]
    color = colors[color_offset % len(colors)]

    # Plot links with transparency
    ax.plot([x0, x1], [z0, z1], color=color, alpha=alpha, linewidth=2)
    ax.plot([x1, x2], [z1, z2], color=color, alpha=alpha, linewidth=2)
    ax.plot([x2, x3], [z2, z3], color=color, alpha=alpha, linewidth=2)

    # Plot joints with transparency
    ax.plot(
        [x0, x1, x2, x3], [z0, z1, z2, z3], "o", color=color, alpha=alpha, markersize=4
    )

    return x3, z3  # Return end-effector position


def plot_trajectory(pitch_deg, color_offset, l0, l1, l2, l3, ax, num_steps=15):
    """Plot trajectory for a given pitch angle."""
    # Convert pitch to the correct phi value
    # pitch = 0° means pointing vertically upward (phi = pi/2)
    # pitch > 0° means tilted forward from vertical
    pitch_rad = np.radians(pitch_deg)
    phi = np.pi / 2 + pitch_rad  # pi/2 is vertical, positive pitch tilts forward

    z_values = np.linspace(0.1, 0.7, num_steps)

    end_effector_x = []
    end_effector_z = []

    # Use previous solution as initial guess for next step
    initial_guess = np.array([0.0, 0.0, pitch_rad])  # Start with a reasonable guess

    for i, zd in enumerate(z_values):
        try:
            solution = constrained_ik(0.0, zd, phi, l0, l1, l2, l3, initial_guess)

            # Plot the mechanism configuration
            x_end, z_end = plot_mechanism_transparent(
                *solution, l0, l1, l2, l3, ax, alpha=0.4, color_offset=color_offset
            )

            end_effector_x.append(x_end)
            end_effector_z.append(z_end)

            # Use current solution as initial guess for next iteration
            initial_guess = solution

        except Exception as e:
            print(f"Failed at z={zd:.2f} for pitch {pitch_deg}°: {e}")
            break

    # Plot end-effector trajectory
    colors = ["blue", "green", "red"]
    color = colors[color_offset % len(colors)]
    ax.plot(
        end_effector_x,
        end_effector_z,
        "o-",
        color=color,
        linewidth=3,
        markersize=6,
        alpha=0.8,
        label=f"End-effector (pitch={pitch_deg}°)",
    )

    return end_effector_x, end_effector_z


def main():
    # Link lengths
    l0, l1, l2, l3 = 0.05, 0.4, 0.4, 0.05

    # Create three separate subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Test three different pitch angles
    pitch_angles = [0, 15, 30]  # degrees

    print("=== Trajectory Analysis ===")

    for i, pitch_deg in enumerate(pitch_angles):
        print(f"\nComputing trajectory for pitch = {pitch_deg}°")
        ax = axes[i]

        end_x, end_z = plot_trajectory(pitch_deg, i, l0, l1, l2, l3, ax)

        print(
            f"End-effector range: x=[{min(end_x):.3f}, {max(end_x):.3f}], z=[{min(end_z):.3f}, {max(end_z):.3f}]"
        )

        # Customize each subplot
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Z (m)", fontsize=12)
        ax.set_title(f"Pitch Angle: {pitch_deg}°\n(15 configurations)", fontsize=14)
        ax.legend(fontsize=10)

        # Add a vertical reference line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

        # Set reasonable limits
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

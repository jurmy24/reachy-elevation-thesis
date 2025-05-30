import numpy as np
import matplotlib.pyplot as plt
from ik import analytical_ik, numerical_ik


def plot_mechanism(theta1, theta2, theta3, l0, l1, l2, l3, ax=None, color_offset=0):
    """Plot the 3-link mechanism in its current configuration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Define colors for different pitch angles
    colors = ["blue", "green", "red"]
    color = colors[color_offset % len(colors)]
    alpha = 0.3

    # Calculate joint positions (manually because FK only returns end effector position)
    x0, z0 = 0, l0
    x1 = x0 + l1 * np.cos(theta1)
    z1 = z0 + l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    z2 = z1 + l2 * np.sin(theta1 + theta2)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    z3 = z2 + l3 * np.sin(theta1 + theta2 + theta3)

    phi = theta1 + theta2 + theta3
    # print(f"phi: {phi}")

    # Plot links
    ax.plot([x0, x1], [z0, z1], color=color, alpha=alpha, linewidth=2)
    ax.plot([x1, x2], [z1, z2], color=color, alpha=alpha, linewidth=2)
    ax.plot([x2, x3], [z2, z3], color=color, alpha=alpha, linewidth=2)

    # Plot joints
    ax.plot(
        [x0, x1, x2, x3], [z0, z1, z2, z3], "o", color=color, alpha=alpha, markersize=3
    )

    return x3, z3


def plot_trajectory(pitch_rad, color_offset, l0, l1, l2, l3, ax, num_steps=7):
    """Plot trajectory for a given pitch angle."""
    # Convert pitch to the correct phi value
    # pitch = 0° means pointing vertically upward (phi = pi/2)
    # pitch > 0° means tilted forward from vertical
    phi = np.pi / 2 - pitch_rad  # pi/2 is vertical, positive pitch tilts forward

    z_values = np.linspace(0.3541, 0.9701, num_steps)

    end_effector_x = []
    end_effector_z = []

    # Use previous solution as initial guess for next step
    initial_guess = np.array([0.0, 0.0, pitch_rad])  # Start with a reasonable guess

    for i, zd in enumerate(z_values):
        try:
            solution = numerical_ik(0.0, zd, phi, l0, l1, l2, l3, initial_guess)

            # Plot the mechanism configuration
            x_end, z_end = plot_mechanism(
                *solution, l0, l1, l2, l3, ax, color_offset=color_offset
            )

            end_effector_x.append(x_end)
            end_effector_z.append(z_end)

            # Use current solution as initial guess for next iteration
            initial_guess = solution

        except Exception as e:
            print(
                f"Failed at z={zd:.2f} for pitch {int(round(np.rad2deg(pitch_rad)))}°: {e}"
            )
            break

    # Plot end-effector trajectory
    # colors = ["blue", "green", "red"]
    # color = colors[color_offset % len(colors)]
    # ax.plot(
    #     end_effector_x,
    #     end_effector_z,
    #     "o-",
    #     color=color,
    #     linewidth=3,
    #     markersize=3,
    #     alpha=0.8,
    #     label=f"End-effector (pitch={int(round(np.rad2deg(pitch_rad)))}°)",
    # )

    return end_effector_x, end_effector_z


def main():
    # Link lengths
    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05

    # Create three separate subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Test three different pitch angles
    pitch_angles = np.radians([0, 15, 30])  # degrees

    print("=== Trajectory Analysis ===")

    for i, pitch_rad in enumerate(pitch_angles):
        print(f"\nComputing trajectory for pitch = {pitch_rad}°")
        ax = axes[i]

        end_x, end_z = plot_trajectory(pitch_rad, i, l0, l1, l2, l3, ax)

        print(
            f"End-effector range: x=[{min(end_x):.3f}, {max(end_x):.3f}], z=[{min(end_z):.3f}, {max(end_z):.3f}]"
        )

        # Customize each subplot
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Z (m)", fontsize=12)
        ax.set_title(
            f"Pitch Angle: {int(round(np.rad2deg(pitch_rad)))}°\n(7 configurations)",
            fontsize=14,
        )
        ax.legend(fontsize=10)

        # Add a vertical reference line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

        # Set reasonable limits
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

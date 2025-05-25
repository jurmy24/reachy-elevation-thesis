import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# --- Units ---
# 725px = 200mm, so 1px = 200/725 mm = 0.2759mm = 0.0002759m
px_to_m = 200 / 725 / 1000  # conversion factor from pixels to meters
m_to_px = 1 / px_to_m  # conversion factor from meters to pixels

# Image dimensions in meters
img_width_px = 2000
img_height_px = 2000
img_width_m = img_width_px * px_to_m
img_height_m = img_height_px * px_to_m

# Define plot boundaries in meters
x_min, x_max = -img_width_m / 2, img_width_m / 2
y_min, y_max = -img_height_m / 2, img_height_m / 2

# Plot limits (expanded beyond image dimensions)
plot_margin = 1  # meters of extra space around the image
plot_x_min, plot_x_max = x_min - plot_margin, x_max + plot_margin
plot_y_min, plot_y_max = y_min - plot_margin, y_max + plot_margin

# Constants
z_base = 0.3541  # meters
l_torso = 0.245  # meters
l_ext = 0.05  # meters
l_arm = 0.28 + 0.28 + 0.11  # meters
x_origin = 0.0  # meters
y_origin = 0.0  # meters


# Note that pitch is in degrees
def contour_per_pitch(pitch: float):
    # Convert pitch to radians
    pitch_rad = np.deg2rad(pitch)

    # Get shoulder positions
    right_shoulder_x, right_shoulder_y = (
        0.03527 + (l_torso + l_ext) * np.sin(pitch_rad),
        -0.2,
    )
    left_shoulder_x, left_shoulder_y = right_shoulder_x, 0.2

    # Get vertical position of the shoulders
    shoulder_z = z_base + l_torso
    shoulder_z = shoulder_z + (l_torso + l_ext) * (np.cos(pitch_rad) - 1)

    # Get radius of contour using Pythagorean theorem
    r_squared = l_arm**2 - shoulder_z**2
    if r_squared < 0:
        print(
            f"Warning: At pitch {pitch}°, arm cannot reach floor (shoulder_z = {shoulder_z}m, l_arm = {l_arm}m)"
        )
        r = 0  # No reachable area
    else:
        r = np.sqrt(r_squared)

    # Generate circular contour points
    angles = np.linspace(0, 2 * np.pi, 100)

    # Right arm contour
    right_x = right_shoulder_x + r * np.cos(angles)
    right_y = right_shoulder_y + r * np.sin(angles)

    # Left arm contour
    left_x = left_shoulder_x + r * np.cos(angles)
    left_y = left_shoulder_y + r * np.sin(angles)
    return {
        "right_arm": (right_x, right_y),
        "left_arm": (left_x, left_y),
        "right_shoulder": (right_shoulder_x, right_shoulder_y),
        "left_shoulder": (left_shoulder_x, left_shoulder_y),
        "radius": r,
        "shoulder_z": shoulder_z,
    }


# --- Load image and prepare plot ---
try:
    img = imread("../../assets/reachy-top-200x200.png")
    # Rotate image 180 degrees
    img = np.rot90(img, k=2)
    has_image = True
except FileNotFoundError:
    has_image = False
    print("Image file not found, proceeding without background image.")


# Calculate figure size in inches
# But keep it a reasonable size for display
inches_per_meter = 39.37  # 1 meter = 39.37 inches
target_width = 10
scale_factor = target_width / (img_width_m * inches_per_meter)
fig_width = target_width
fig_height = img_height_m * inches_per_meter * scale_factor
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

if has_image:
    # x-axis: forward (positive to the top of the image)
    # y-axis: left (positive to the left of the image)

    # Shift the image position relative to the origin since the shoulders are extended
    shift_x = 0.03527
    shifted_x_min = x_min + shift_x
    shifted_x_max = x_max + shift_x
    shifted_y_min = y_min
    shifted_y_max = y_max

    ax.imshow(
        img,
        extent=[shifted_y_min, shifted_y_max, shifted_x_min, shifted_x_max],
        origin="lower",
        alpha=1,
    )

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

# --- Define pitch angles and colors ---
pitches = [-30, -15, 0, 15, 30]
colors = plt.cm.viridis(np.linspace(0, 1, len(pitches)))

# --- Plot contours for each pitch ---
for pitch, c in zip(pitches, colors):
    contours = contour_per_pitch(pitch)

    if contours["radius"] > 0:
        # Calculate torso position based on pitch
        pitch_rad = np.deg2rad(pitch)
        torso_x = (l_torso + l_ext) * np.sin(pitch_rad)
        torso_y = 0

        # Plot torso position
        ax.plot([0, torso_y], [0, torso_x], "k-", linewidth=1, alpha=0.5)
        ax.plot(torso_y, torso_x, "o", color=c, markersize=4)

        # Plot shoulder positions
        rsx, rsy = contours["right_shoulder"]
        lsx, lsy = contours["left_shoulder"]
        ax.plot(rsy, rsx, "o", color=c, markersize=3)
        ax.plot(lsy, lsx, "o", color=c, markersize=3)

        # Plot connection from torso to shoulders
        ax.plot([torso_y, rsy], [torso_x, rsx], "-", color=c, linewidth=0.5, alpha=0.7)
        ax.plot([torso_y, lsy], [torso_x, lsx], "-", color=c, linewidth=0.5, alpha=0.7)

        # Plot arm contour
        right_x, right_y = contours["right_arm"]
        left_x, left_y = contours["left_arm"]

        # For pitch = 0, fill the contour with a lighter color
        if pitch == 0:
            lighter_c = c.copy()
            # Set transparency for the fill
            lighter_c[3] = 0.5

            # Fill arm contour
            ax.fill(right_y, right_x, color=lighter_c, alpha=0.15)
            ax.fill(left_y, left_x, color=lighter_c, alpha=0.15)

            # Draw the contour lines with full opacity
            ax.plot(
                right_y,
                right_x,
                color=c,
                linestyle="-",
                linewidth=2,
                label=f"Pitch {pitch}°",
            )
            ax.plot(left_y, left_x, color=c, linestyle="-", linewidth=2)
        else:
            # Normal line plotting for other pitches
            ax.plot(
                right_y,
                right_x,
                color=c,
                linestyle="-",
                linewidth=2,
                label=f"Pitch {pitch}°",
            )
            ax.plot(left_y, left_x, color=c, linestyle="-", linewidth=2)

ax.set_xlabel("y (m)")
ax.set_ylabel("x (m)")
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")  # Ensure equal aspect ratio
ax.grid(True, linestyle="--", alpha=0.5)

# Swap x and y for the expanded limits
expanded_x_min = y_min - 0.4
expanded_x_max = y_max + 0.4
expanded_y_min = x_min - 0.4
expanded_y_max = x_max + 0.4

ax.set_xlim(expanded_x_min, expanded_x_max)
ax.set_ylim(expanded_y_min, expanded_y_max)

plt.tight_layout()
plt.show()

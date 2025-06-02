import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import minimize_scalar

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
# z_base = 0.3542  # meters
l_torso = 0.245  # meters
l_ext = 0.05  # meters
l_arm = 0.28 + 0.28 + 0.11  # meters
x_origin = 0.0  # meters
y_origin = 0.0  # meters

# --- Optimization parameters ---
pitch_for_optimization = 0  # degrees
target_radius = 0.3  # meters
target_area = 0.0  # m²
if 0.2 >= target_radius:
    # The circles don't overlap
    A = 2 * np.pi * target_radius**2
else:
    # The circles overlap
    print(f"Circles overlap, r={target_radius:.4f}m")
    zeta = np.arccos(0.2 / target_radius)
    target_area = 2 * np.pi * target_radius**2 - 2 * target_radius**2 * (
        zeta - 0.5 * np.sin(2 * zeta)
    )
print(f"Target area: {target_area:.4f} m²")


# Note that pitch is in degrees
def contour_per_pitch(pitch: float, custom_z_base=None):
    # Use custom z_base if provided, otherwise use default
    z_base_val = custom_z_base

    # Convert pitch to radians
    pitch_rad = np.deg2rad(pitch)

    # Get shoulder positions
    # For negative pitch, the shoulders move backward (negative x)
    right_shoulder_x, right_shoulder_y = (
        0.03527 + (l_torso + l_ext) * np.sin(pitch_rad),
        -0.2,  # Right shoulder is negative y (to the right of center)
    )
    left_shoulder_x, left_shoulder_y = right_shoulder_x, 0.2

    # Get vertical position of the shoulders
    shoulder_z = z_base_val + l_torso
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


def calculate_workspace_area(pitch=0, custom_z_base=None):
    """
    Calculate the area of the workspace, accounting for overlap between the two arms.
    Uses a Monte Carlo approach to find the area.

    Parameters:
    -----------
    pitch : float
        The pitch angle in degrees
    custom_z_base : float or None
        Custom z_base value. If None, uses the default z_base

    Returns:
    --------
    float : The area of the workspace in square meters
    """
    contours = contour_per_pitch(pitch, custom_z_base)

    if contours["radius"] <= 0:
        return 0.0  # No reachable area

    # Extract radius
    r = contours["radius"]

    A = 0.0
    if 0.2 >= r:
        # The circles don't overlap
        A = 2 * np.pi * r**2
    else:
        # The circles overlap
        zeta = np.arccos(0.2 / r)
        A = 2 * np.pi * r**2 - 2 * r**2 * (zeta - 0.5 * np.sin(2 * zeta))

    return A


# This is a numerical solution
def find_z_base_for_target_area(target_area, pitch=0):
    """
    Find the z_base value that gives a workspace with the target area.

    Parameters:
    -----------
    target_area : float
        The desired workspace area in square meters
    pitch : float
        The pitch angle in degrees

    Returns:
    --------
    float : The z_base value that gives the target area
    float : The actual area achieved with the returned z_base
    """

    def objective(z_val):
        # Calculate the difference between the achieved area and the target area
        area = calculate_workspace_area(pitch=pitch, custom_z_base=z_val)
        return abs(area - target_area)

    min_z_base = 0.2401  # The height of the holonomic base is the minimum
    max_z_base = l_arm - l_torso

    result = minimize_scalar(
        objective, bounds=(min_z_base, max_z_base), method="bounded"
    )

    optimal_z_base = result.x
    achieved_area = calculate_workspace_area(pitch=pitch, custom_z_base=optimal_z_base)

    return optimal_z_base, achieved_area


# --- Load image and prepare plot ---
try:
    img = imread("../../assets/reachy-top-200x200.png")
    # Rotate image 180 degrees
    img = np.rot90(img, k=2)
    has_image = True
except FileNotFoundError:
    has_image = False
    print("Image file not found, proceeding without background image.")


# Calculate figure size in inches to match the image scale
inches_per_meter = 39.37  # 1 meter = 39.37 inches
target_width = 10  # inches
scale_factor = target_width / (img_width_m * inches_per_meter)
fig_width = target_width
fig_height = img_height_m * inches_per_meter * scale_factor
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

if has_image:
    # When displaying the image, we need to make sure the coordinate system aligns
    # x-axis: forward (positive to the top of the image) -> now y-axis
    # y-axis: left (positive to the left of the image) -> now x-axis

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

    # Add axis lines to show the origin
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)


# Find z_base for target area
optimal_z_base, achieved_area = find_z_base_for_target_area(
    target_area, pitch=pitch_for_optimization
)
print(f"\n--- Optimization Result ---")
print(f"Target area: {target_area:.4f} m²")
print(f"Optimal z_base: {optimal_z_base:.4f} m")
print(f"Achieved area: {achieved_area:.4f} m²")

# Create a new figure for the workspace area vs z_base plot
plt.figure(figsize=(10, 6))

# Generate z_base values to plot
z_base_values = np.linspace(0.2401, l_arm - l_torso, 100)
area_values = [
    calculate_workspace_area(pitch=pitch_for_optimization, custom_z_base=z)
    for z in z_base_values
]

# Plot the relationship
plt.plot(z_base_values, area_values, "b-", label="Workspace Area")
plt.axhline(
    y=target_area,
    color="r",
    linestyle="--",
    label=f"Target Area ({target_area:.4f} m²)",
)
plt.axvline(
    x=optimal_z_base,
    color="g",
    linestyle="--",
    label=rf"Optimal $z_{{ee}}$ ({optimal_z_base:.4f} m)",
)

# Add labels and title
plt.xlabel(r"$z_{ee}$ (m)")
plt.ylabel("Workspace Area (m²)")
# plt.title("Workspace Area vs z_base")
plt.grid(True)
plt.legend()

# Show all plots
plt.show()

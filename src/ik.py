import numpy as np
import math


def forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3):
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
    phi = theta1 + theta2 + theta3
    return np.array([x, z, phi])


def analytical_ik(xd, zd, phi, l0, l1, l2, l3):
    z_prime = zd - l0
    xw = xd - l3 * np.cos(phi)
    zw = z_prime - l3 * np.sin(phi)

    D = (xw**2 + zw**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(D) > 1:
        raise ValueError("Target is unreachable.")

    theta2 = math.atan2(math.sqrt(1 - D**2), D)
    theta1 = math.atan2(zw, xw) - math.atan2(
        l2 * math.sin(theta2), l1 + l2 * math.cos(theta2)
    )
    theta3 = phi - theta1 - theta2

    return np.array([theta1, theta2, theta3])


from scipy.optimize import least_squares


def numerical_ik(xd, zd, phi, l0, l1, l2, l3, initial_guess):
    def objective(thetas):
        theta1, theta2, theta3 = thetas
        xz_phi = forward_kinematics_direct(theta1, theta2, theta3, l0, l1, l2, l3)
        return xz_phi - np.array([xd, zd, phi])

    # Set joint limits in radians (adjust as needed)
    lower_bounds = np.radians([-180, -180, -180])
    upper_bounds = np.radians([180, 180, 180])

    result = least_squares(
        objective,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    if not result.success:
        raise RuntimeError("Numerical IK did not converge.")
    return result.x


# ---- Comparison ----
# Desired end-effector position and orientation
xd, zd = 0.0, 0.6
phi = math.pi / 6  # 30 degrees

# Link lengths
l0, l1, l2, l3 = 0.05, 0.4, 0.4, 0.05

# Compute IK solutions
analytical_solution = analytical_ik(xd, zd, phi, l0, l1, l2, l3)
numerical_solution = numerical_ik(
    xd, zd, phi, l0, l1, l2, l3, initial_guess=np.array([0.0, 0.0, 0.0])
)

# Compute FK from both solutions
fk_analytical = forward_kinematics_direct(*analytical_solution, l0, l1, l2, l3)
fk_numerical = forward_kinematics_direct(*numerical_solution, l0, l1, l2, l3)


# ---- Output ----
def print_solution(name, solution):
    deg = np.degrees(solution)
    print(f"{name} Joint Angles (rad): {solution}")
    print(f"{name} Joint Angles (deg): {deg}")


print("=== Inverse Kinematics Comparison ===")
print_solution("Analytical", analytical_solution)
print()
print_solution("Numerical", numerical_solution)
print()

print("=== Forward Kinematics Check ===")
print(f"Desired:     [x={xd:.4f}, z={zd:.4f}, phi={phi:.4f}]")
print(f"From Analytic FK: {fk_analytical}")
print(f"From Numeric FK:  {fk_numerical}")
print()

# Error check
error_analytic = np.linalg.norm(fk_analytical - np.array([xd, zd, phi]))
error_numeric = np.linalg.norm(fk_numerical - np.array([xd, zd, phi]))

print("=== Errors ===")
print(f"Analytical FK Error: {error_analytic:.6e}")
print(f"Numerical FK Error:  {error_numeric:.6e}")

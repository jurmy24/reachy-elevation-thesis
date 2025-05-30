import numpy as np
import math
from fk import forward_kinematics_direct
from scipy.optimize import minimize


def analytical_ik(xd, zd, phi, l0, l1, l2, l3, initial_guess=None):
    z_prime = zd - l0
    xw = xd - l3 * np.cos(phi)
    zw = z_prime - l3 * np.sin(phi)

    D = (xw**2 + zw**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(D) > 1:
        # Small epsilon for floating point comparisons
        epsilon = 1e-9
        if abs(D) - 1 < epsilon:
            D = np.sign(D)  # Clamp to +/- 1 if very close
        else:
            raise ValueError(f"Target is unreachable. D = {D}")

    # Ensure argument for sqrt is non-negative
    sqrt_arg = 1 - D**2
    if sqrt_arg < 0:
        sqrt_arg = 0  # Clamp to zero if slightly negative due to precision

    theta2 = math.atan2(math.sqrt(sqrt_arg), D)
    theta1 = math.atan2(zw, xw) - math.atan2(
        l2 * math.sin(theta2), l1 + l2 * math.cos(theta2)
    )
    theta3 = phi - theta1 - theta2

    # Return as a flat numpy array, just like numerical_ik and constrained_ik
    return np.array([theta1, theta2, theta3])


def numerical_ik(xd, zd, phi, l0, l1, l2, l3, initial_guess):
    def objective(thetas):
        theta1, theta2, theta3 = thetas
        current_pos_vec, current_phi_scalar = forward_kinematics_direct(
            theta1, theta2, theta3, l0, l1, l2, l3
        )
        error_x = current_pos_vec[0] - xd
        error_z = current_pos_vec[2] - zd

        delta_phi = current_phi_scalar - phi
        error_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

        # Return a vector of errors for least_squares
        return error_x**2 + error_z**2 + error_phi**2

    # Set joint limits in radians
    # bounds = [(0, np.pi) for _ in range(3)]

    bounds = [(np.pi / 2 + 0.01, 3 * np.pi / 2), (-np.pi, np.pi), (-np.pi, np.pi)]

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        options={"ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError(
            f"Numerical IK did not converge. Status: {result.status}, Message: {result.message}"
        )
    return result.x


# ---- Output ----
def print_solution(name, solution):
    deg = np.degrees(solution)
    print(f"{name} Joint Angles (rad): {solution}")
    print(f"{name} Joint Angles (deg): {deg}")


if __name__ == "__main__":

    # ---- Comparison ----
    # Desired end-effector position and orientation
    xd, zd = 0.0, 0.6
    # phi = math.pi / 6  # 30 degrees
    phi = 0.0  # Target phi = 0 radians

    # Link lengths
    l0, l1, l2, l3 = 0.2901, 0.4, 0.4, 0.05

    # Compute IK solutions
    analytical_solution = analytical_ik(xd, zd, phi, l0, l1, l2, l3)
    numerical_solution = numerical_ik(
        xd, zd, phi, l0, l1, l2, l3, initial_guess=np.array([0.0, 0.0, 0.0])
    )

    # Compute FK from both solutions
    fk_analytical_pos, analytical_phi = forward_kinematics_direct(
        *analytical_solution, l0, l1, l2, l3
    )
    fk_numerical_pos, numerical_phi = forward_kinematics_direct(
        *numerical_solution, l0, l1, l2, l3
    )
    print("=== Inverse Kinematics Comparison ===")
    print_solution("Analytical", analytical_solution)
    print()
    print_solution("Numerical", numerical_solution)
    print()

    print("=== Forward Kinematics Check ===")
    print(f"Desired:     [x={xd:.4f}, z={zd:.4f}, phi={phi:.4f}]")
    print(
        f"From Analytic FK: [x={fk_analytical_pos[0]:.4f}, z={fk_analytical_pos[2]:.4f}, phi={analytical_phi:.4f}]"
    )
    print(
        f"From Numeric FK:  [x={fk_numerical_pos[0]:.4f}, z={fk_numerical_pos[2]:.4f}, phi={numerical_phi:.4f}]"
    )
    print()

    # Error check
    # Analytical Error
    error_x_analytic = fk_analytical_pos[0] - xd
    error_z_analytic = fk_analytical_pos[2] - zd
    delta_phi_analytic = analytical_phi - phi
    error_phi_analytic = math.atan2(
        math.sin(delta_phi_analytic), math.cos(delta_phi_analytic)
    )
    error_analytic = np.linalg.norm(
        [error_x_analytic, error_z_analytic, error_phi_analytic]
    )

    # Numerical Error
    error_x_numeric = fk_numerical_pos[0] - xd
    error_z_numeric = fk_numerical_pos[2] - zd
    delta_phi_numeric = numerical_phi - phi
    error_phi_numeric = math.atan2(
        math.sin(delta_phi_numeric), math.cos(delta_phi_numeric)
    )
    error_numeric = np.linalg.norm(
        [error_x_numeric, error_z_numeric, error_phi_numeric]
    )

    print("=== Errors ===")
    print(f"Analytical FK Error: {error_analytic:.6e}")
    print(f"Numerical FK Error:  {error_numeric:.6e}")

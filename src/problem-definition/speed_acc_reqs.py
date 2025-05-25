# Bang-Bang Motion Profile Calculator


def compute_bang_bang_params(s_max, T_total, v_max=None, a_max=None):
    """
    Compute the missing parameter (a_max or v_max) based on trapezoidal motion:
    T_total = s_max / v_max + v_max / a_max
    """
    if v_max and not a_max:
        T_const = s_max / v_max
        T_acc = T_total - T_const
        if T_acc <= 0:
            raise ValueError("Given v_max is too low to reach destination in time.")
        a_max = v_max / T_acc
        return v_max, a_max

    elif a_max and not v_max:
        # Solve: T = s/v + v/a => T*a = s*a/v + v
        # => T*a*v = s*a + v^2 => v^2 - T*a*v + s*a = 0 (quadratic in v)
        import math

        A = 1
        B = -T_total * a_max
        C = s_max * a_max
        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            raise ValueError("No valid solution with given a_max and T_total.")

        v_max_1 = (-B + math.sqrt(discriminant)) / (2 * A)
        v_max_2 = (-B - math.sqrt(discriminant)) / (2 * A)
        v_max = max(v_max_1, v_max_2)
        return v_max, a_max

    else:
        raise ValueError("Specify exactly one of v_max or a_max.")


# === Example Usage ===
s_max = 0.616  # meters
T_total = 10.0  # seconds

# Option 1: Set v_max and compute a_max
v_max_input = 0.1  # m/s
# a_max_input = 0.025  # m/s^2
v_max, a_max = compute_bang_bang_params(s_max, T_total, a_max=None, v_max=v_max_input)

# Option 2: Uncomment to use a_max instead
# a_max_input = 0.025  # m/s^2
# v_max, a_max = compute_bang_bang_params(s_max, T_total, a_max=a_max_input)

# === Output ===
print("=== Bang-Bang Profile Parameters ===")
print(f"Distance to travel:     {s_max} m")
print(f"Total time allowed:     {T_total} s")
print(f"Computed v_max:         {v_max:.4f} m/s")
print(f"Computed a_max:         {a_max:.4f} m/s²")
print(f"Single acceleration phase: {v_max / a_max:.4f} s")
print(f"Total accel/decel time:   {2 * (v_max / a_max):.4f} s")
print(f"Constant velocity time: {T_total - 2 * (v_max / a_max):.4f} s")
print(f"Average velocity:       {s_max / T_total:.4f} m/s")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


class DraggableKinematicChain:
    def __init__(self):
        # Fixed ground joints
        self.fixed_joints = {
            "J_a1": np.array([0.1, 0.30]),
            "J_a2": np.array([0.0, 0.25]),
        }
        # Link lengths
        self.L = {
            "l_a1": 0.35,  # J_a1–J_k1
            "l_k1": 0.35,  # J_k1–J_h1
            "l_a2": 0.35,  # J_a2–J_k2
            "l_kk": np.linalg.norm(
                self.fixed_joints["J_a1"] - self.fixed_joints["J_a2"]
            ),  # J_k2–J_k1
            "l_k2": 0.35,  # J_k2–J_h2
            "l_hh": np.linalg.norm(
                self.fixed_joints["J_a1"] - self.fixed_joints["J_a2"]
            ),  # J_h1–J_h3
            "l_pivot": 0.05,  # J_h2–J_h3
        }
        self.joint_names = ["J_a1", "J_a2", "J_k1", "J_k2", "J_h1", "J_h2", "J_h3"]
        self.links = [
            ("J_a1", "J_k1", "l_a1"),
            ("J_k1", "J_h1", "l_k1"),
            ("J_a2", "J_k2", "l_a2"),
            ("J_k2", "J_k1", "l_kk"),
            ("J_k2", "J_h2", "l_k2"),
            ("J_h1", "J_h3", "l_hh"),
            ("J_h2", "J_h3", "l_pivot"),
        ]
        # Initial guess for floating joints (x, y for each)
        self.floating_joints_init = np.array(
            [
                -0.10,
                0.50,  # J_k1
                -0.15,
                0.45,  # J_k2
                0.1,
                0.75,  # J_h1
                -0.05,
                0.65,  # J_h2
                -0.05,
                0.70,  # J_h3
            ]
        )
        self.floating_joint_names = ["J_k1", "J_k2", "J_h1", "J_h2", "J_h3"]
        self.floating_joint_indices = {
            "J_k1": (0, 1),
            "J_k2": (2, 3),
            "J_h1": (4, 5),
            "J_h2": (6, 7),
            "J_h3": (8, 9),
        }
        self.solve_positions()
        self.dragging_joint = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        plt.show()

    def get_joint_pos(self, x, name):
        if name in self.fixed_joints:
            return self.fixed_joints[name]
        idx = self.floating_joint_indices[name]
        return np.array([x[idx[0]], x[idx[1]]])

    def constraint_equations(self, x, fixed_name=None, fixed_pos=None):
        eqs = []
        for j1, j2, lname in self.links:
            p1 = self.get_joint_pos(x, j1) if j1 != fixed_name else fixed_pos
            p2 = self.get_joint_pos(x, j2) if j2 != fixed_name else fixed_pos
            eqs.append(np.linalg.norm(p1 - p2) - self.L[lname])
        return np.array(eqs)

    def solve_positions(self, fixed_name=None, fixed_pos=None):
        def fun(x):
            return self.constraint_equations(x, fixed_name, fixed_pos)

        res = least_squares(
            fun,
            self.floating_joints_init,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
        )
        self.floating_joints = res.x
        self.floating_joints_init = self.floating_joints.copy()
        self.closure_error = np.linalg.norm(
            self.constraint_equations(self.floating_joints)
        )
        self.update_joint_positions()

    def update_joint_positions(self):
        idx = self.floating_joint_indices
        self.joint_positions = {
            "J_a1": self.fixed_joints["J_a1"],
            "J_a2": self.fixed_joints["J_a2"],
            "J_k1": self.floating_joints[idx["J_k1"][0] : idx["J_k1"][1] + 1],
            "J_k2": self.floating_joints[idx["J_k2"][0] : idx["J_k2"][1] + 1],
            "J_h1": self.floating_joints[idx["J_h1"][0] : idx["J_h1"][1] + 1],
            "J_h2": self.floating_joints[idx["J_h2"][0] : idx["J_h2"][1] + 1],
            "J_h3": self.floating_joints[idx["J_h3"][0] : idx["J_h3"][1] + 1],
        }

    def setup_plot(self):
        self.ax.set_xlim(-0.4, 0.4)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Draggable Closed-Loop Kinematic Chain")
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-0.4, 0.4)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Draggable Closed-Loop Kinematic Chain")
        colors = ["red", "blue", "green", "orange", "purple", "brown", "gray"]
        for i, (j1, j2, lname) in enumerate(self.links):
            p1 = self.joint_positions[j1]
            p2 = self.joint_positions[j2]
            self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=colors[i % len(colors)],
                linewidth=4,
                label=f"{lname}: {self.L[lname]:.2f}m",
            )
        for name, pos in self.joint_positions.items():
            if name.startswith("J_a"):
                self.ax.plot(pos[0], pos[1], "ks", markersize=10)
            else:
                self.ax.plot(pos[0], pos[1], "ro", markersize=8, picker=5)
            self.ax.annotate(
                name,
                (pos[0], pos[1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"),
            )
        # Draw ghost marker if dragging
        if (
            hasattr(self, "ghost_pos")
            and self.ghost_pos is not None
            and self.dragging_joint is not None
        ):
            self.ax.plot(
                self.ghost_pos[0],
                self.ghost_pos[1],
                "bo",
                markersize=12,
                alpha=0.4,
                label="Ghost",
            )
        info_text = f"Closure Error: {self.closure_error:.4e} m"
        self.ax.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Find closest floating joint
        min_dist = 0.02
        for name in self.floating_joint_names:
            pos = self.joint_positions[name]
            dist = np.linalg.norm([event.xdata - pos[0], event.ydata - pos[1]])
            if dist < min_dist:
                self.dragging_joint = name
                self.ghost_pos = np.array([event.xdata, event.ydata])
                self.update_plot()
                break

    def on_motion(self, event):
        if self.dragging_joint is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Just update the ghost marker position for feedback
        self.ghost_pos = np.array([event.xdata, event.ydata])
        self.update_plot()

    def on_release(self, event):
        if self.dragging_joint is None or event.inaxes != self.ax:
            self.dragging_joint = None
            self.ghost_pos = None
            self.update_plot()
            return
        if event.xdata is None or event.ydata is None:
            self.dragging_joint = None
            self.ghost_pos = None
            self.update_plot()
            return
        # Use the mouse position as the initial guess for the dragged joint, but let the solver enforce all link constraints
        idx = self.floating_joint_indices[self.dragging_joint]
        new_guess = self.floating_joints_init.copy()
        new_guess[idx[0]] = event.xdata
        new_guess[idx[1]] = event.ydata
        self.floating_joints_init = new_guess
        fixed_pos = np.array([event.xdata, event.ydata])
        self.solve_positions(fixed_name=self.dragging_joint, fixed_pos=fixed_pos)
        self.dragging_joint = None
        self.ghost_pos = None
        self.update_plot()


if __name__ == "__main__":
    print("Draggable Closed-Loop Hip Mechanism Interactive Visualization")
    DraggableKinematicChain()

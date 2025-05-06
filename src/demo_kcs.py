"""
This module demonstrates the creation and visualization of various
kinematic couplings using the `panel_kc_errors_v1` module.

Its process was designed and implemented by Trevor K. Carter as an
employee of the CMR at BYU for a lab contract with NASA.
It was last modified on 05/06/2025.
"""

from panel_kc_errors_v1 import KinematicCoupling, MaxwellKinematicCoupling, ShortArmKinematicCoupling, SideArm3BallKinematicCoupling

import numpy as np
import matplotlib.pyplot as plt

def demo_maxwell_creation():
    big_kc = MaxwellKinematicCoupling("big_kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 5, -5)
    short_kc = MaxwellKinematicCoupling("short_kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 1, -1)
    small_kc = MaxwellKinematicCoupling("small_kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 4, 2, -2)

    misalignment_trans = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    misalignment_rot = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    np.random.seed(42)
    misalignment_rand = 2 * np.random.rand(6) - 1
    
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(9, 8))
    fig.suptitle("Maxwell Kinematic Coupling Mounts (Upward)", fontsize=16)  # Add figure title
    view_angle = (15, 75)  # Set a consistent view angle for all subplots

    def set_axis_limits(ax):
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_trans, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.set_title("Big Maxwell Translational Misalignment", fontsize=8)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_rot, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Big Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_rand, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Big Maxwell Random Misalignment", fontsize=8)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_trans, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Short Maxwell Translational Misalignment", fontsize=8)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_rot, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Short Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_rand, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Short Maxwell Random Misalignment", fontsize=8)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_trans, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Small Maxwell Translational Misalignment", fontsize=8)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_rot, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Small Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_rand, ax=ax, x_to_z_axis=True)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Small Maxwell Random Misalignment", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()

def demo_kc_maxwell_creation():
    centroid_kc = MaxwellKinematicCoupling("centroid-kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.75, -0.75)
    off_center_kc = MaxwellKinematicCoupling("off-center-kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.75, 0.75, 2)
    orthogonal_kc = MaxwellKinematicCoupling("orthogonal-kc", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.75, -0.75, 0, np.pi/2, "orthogonal")

    misalignment_trans = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    misalignment_rot = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    np.random.seed(42)
    misalignment_rand = 2 * np.random.rand(6) - 1
    
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(9, 8))
    fig.suptitle("Maxwell Kinematic Coupling Hinges", fontsize=16)  # Add figure title
    view_angle = (15, 75)  # Set a consistent view angle for all subplots

    def set_axis_limits(ax):
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    centroid_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.set_title("Centroid Maxwell Translational Misalignment", fontsize=8)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    centroid_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Centroid Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    centroid_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Centroid Maxwell Random Misalignment", fontsize=8)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    off_center_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Maxwell Translational Misalignment", fontsize=8)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    off_center_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    off_center_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Maxwell Random Misalignment", fontsize=8)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    orthogonal_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Orthogonal Maxwell Translational Misalignment", fontsize=8)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    orthogonal_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Orthogonal Maxwell Rotational Misalignment", fontsize=8)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    orthogonal_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Orthogonal Maxwell Random Misalignment", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()


        
def demo_kc_short_arm_creation():
    centered_kc = ShortArmKinematicCoupling("center", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.5, 1, -0.5)
    uncentered_kc = ShortArmKinematicCoupling("off-center", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.5, 1, -0.5, 2)
    steep_kc = ShortArmKinematicCoupling("steep-groove", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 0.5, 1, -0.5, 0, np.pi/3)

    misalignment_trans = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    misalignment_rot = np.array([-0.5, 0.5, 0.5, -0.5, 0.5, -0.5])
    np.random.seed(42)
    misalignment_rand = 2 * np.random.rand(6) - 1
    
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(9, 8))
    fig.suptitle("Short Arm Kinematic Coupling Hinges", fontsize=16)  # Add figure title
    view_angle = (15, 75)  # Set a consistent view angle for all subplots

    def set_axis_limits(ax):
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    centered_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.set_title("Centered Short Arm Translational Misalignment", fontsize=8)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    centered_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Centered Short Arm Rotational Misalignment", fontsize=8)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    centered_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Centered Short Arm Random Misalignment", fontsize=8)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    uncentered_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Short Arm Translational Misalignment", fontsize=8)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    uncentered_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Short Arm Rotational Misalignment", fontsize=8)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    uncentered_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Off-Center Short Arm Random Misalignment", fontsize=8)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    steep_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Steep-Groove Short Arm Translational Misalignment", fontsize=8)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    steep_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Steep-Groove Short Arm Rotational Misalignment", fontsize=8)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    steep_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Steep-Groove Short Arm Random Misalignment", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()


        
def demo_kc_side_arm_creation():
    left_kc = SideArm3BallKinematicCoupling("left", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 4, np.pi/2, 0.5)
    right_kc = SideArm3BallKinematicCoupling("right", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 4, np.pi/2, 0.5, np.pi/2, "right")
    angled_kc = SideArm3BallKinematicCoupling("angled", np.array([0, 0, 0]), np.array([1, 0, 0]), 10, 4, np.pi/3, 0.5)

    misalignment_trans = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    misalignment_rot = np.array([-0.5, 0.5, 0.5, -0.5, 0.5, -0.5])
    np.random.seed(42)
    misalignment_rand = 2 * np.random.rand(6) - 1
    
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(9, 8))
    fig.suptitle("Side Arm Kinematic Coupling Hinges", fontsize=16)  # Add figure title
    view_angle = (45, 75)  # Set a consistent view angle for all subplots

    def set_axis_limits(ax):
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    left_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.set_title("Left Side Arm Translational Misalignment", fontsize=8)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    left_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Left Side Arm Rotational Misalignment", fontsize=8)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    left_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Left Side Arm Random Misalignment", fontsize=8)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    right_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Right Side Arm Translational Misalignment", fontsize=8)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    right_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Right Side Arm Rotational Misalignment", fontsize=8)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    right_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Right Side Arm Random Misalignment", fontsize=8)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    angled_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Angled Side Arm Translational Misalignment", fontsize=8)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    angled_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Angled Side Arm Rotational Misalignment", fontsize=8)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    angled_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    set_axis_limits(ax)
    ax.legend().remove()
    ax.set_title("Angled Side Arm Random Misalignment", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()


if __name__ == "__main__":
    demo_maxwell_creation()
    demo_kc_maxwell_creation()
    demo_kc_short_arm_creation()
    demo_kc_side_arm_creation()
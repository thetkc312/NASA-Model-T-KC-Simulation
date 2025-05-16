"""
Monte Carlo simulation to determine panel misalignment propagation in a linked panel
array, and potential interference between series chains of panels in a full flasher.
I will start by discussing naming conventions for panels and their links, then describe
the set of panels I will use in the simulation.

NAMING CONVENTIONS:
A flasher involves 25 panels linked in an array around a central pentagonal panel, Identified here simply as P.

A typical naming convention for panels is to group them into petals, whose member panels are named as:
A1, A2, A3, B1, B2

I have often identified the panels in a petal that appears immediately counterclockwise
to this petal with a comma after each panel name. Thus, they would be labeled as:
A1, A2, A3, B1, B2,

I have often identified the panels in a petal that appears immediately clockwise to
this petal with an apostraphe after each panel name. Thus, they would be labeled as:
A1' A2' A3' B1' B2'

I will be using a new naming convention for the panels in the simulation. The petal arrangement
does not correspond to the flow of kinematic couplings between panels, whereas this convention does.
Moving radially and outward from the center of the flasher, the panels are labeled as:
S1 S2 S3 S4 S5

The panels in the series chain counterclockwise to this chain are labeled as:
S1- S2- S3- S4- S5-
The panels in the series chain clockwise to this chain are labeled as:
S1+ S2+ S3+ S4+ S5+

MY CHOICE OF PANELS:
I will measure the final pose of all the panels in a series chain. I will build a large table of the
results of this simulation, which will create a file and dynamically add the results. While I will
take basic precautions to avoid issues in the simulation, this will make it robust to memory any
memory issues or bugs that might arise.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from panel_kc_errors_v1 import KinematicCoupling, MaxwellKinematicCoupling, ShortArmKinematicCoupling, SideArm3BallKinematicCoupling
from panel_kc_errors_v1 import Panel, LinkedPanelArray


def long_arm_monte_carlo(series_panel_array, kc_id_list, num_iterations, master_save_location, misalignment_std_dev):
    save_location = os.path.join(master_save_location, "side_arm_monte_carlo")
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    arm_length = 5.0
    kc_p_s1 = SideArm3BallKinematicCoupling("P_S1", np.array([0.000, 0.000, 0], dtype=np.float64), np.array([3.878, 0.000, 0], dtype=np.float64), 11.649, arm_length, 1.616)
    series_panel_array.link_panels(kc_p_s1, "P", "S1")
    kc_s1_s2 = SideArm3BallKinematicCoupling("S1_S2", np.array([4.040, 0.155, 0], dtype=np.float64), np.array([8.026, -2.498, 0], dtype=np.float64), 14.431, arm_length, 1.034)
    series_panel_array.link_panels(kc_s1_s2, "S1", "S2")
    kc_s2_s3 = SideArm3BallKinematicCoupling("S2_S3", np.array([5.695, -3.532, 0], dtype=np.float64), np.array([10.485, -14.937, 0], dtype=np.float64), 12.474, arm_length, 1.619)
    series_panel_array.link_panels(kc_s2_s3, "S2", "S3")
    kc_s3_s4 = SideArm3BallKinematicCoupling("S3_S4", np.array([10.782, -15.782, 0], dtype=np.float64), np.array([8.220, -28.530, 0], dtype=np.float64), 16.333, arm_length, 1.034)
    series_panel_array.link_panels(kc_s3_s4, "S3", "S4")
    kc_s4_s5 = SideArm3BallKinematicCoupling("S4_S5", np.array([2.125, -24.193, 0], dtype=np.float64), np.array([-9.825, -36.397, 0], dtype=np.float64), 14.749, arm_length, 1.606, 0, np.pi/2, "right")
    series_panel_array.link_panels(kc_s4_s5, "S4", "S5")

    series_panel_array.visualize_array_3d()
    plt.show()

    # Run the Monte Carlo simulation
    series_panel_array.run_monte_carlo_simulation(kc_id_list, num_iterations, save_location, misalignment_std_dev)
    # The simulation will save the results in the specified directory

def short_arm_monte_carlo(series_panel_array, kc_id_list, num_iterations, master_save_location, misalignment_std_dev):
    save_location = os.path.join(master_save_location, "short_arm_monte_carlo")
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    top_ball_height = 0.35
    bot_ball_height = -0.25
    arm_length = 0.9
    kc_p_s1 = ShortArmKinematicCoupling("P_S1", np.array([0.000, 0.000, 0], dtype=np.float64), np.array([3.878, 0.000, 0], dtype=np.float64), 11.649, bot_ball_height, arm_length, top_ball_height)
    series_panel_array.link_panels(kc_p_s1, "P", "S1")
    kc_s1_s2 = MaxwellKinematicCoupling("S1_S2", np.array([4.040, 0.155, 0], dtype=np.float64), np.array([8.026, -2.498, 0], dtype=np.float64), 14.431, top_ball_height, bot_ball_height)
    series_panel_array.link_panels(kc_s1_s2, "S1", "S2")
    kc_s2_s3 = ShortArmKinematicCoupling("S2_S3", np.array([5.695, -3.532, 0], dtype=np.float64), np.array([10.485, -14.937, 0], dtype=np.float64), 12.474, bot_ball_height, arm_length, top_ball_height)
    series_panel_array.link_panels(kc_s2_s3, "S2", "S3")
    kc_s3_s4 = MaxwellKinematicCoupling("S3_S4", np.array([10.782, -15.782, 0], dtype=np.float64), np.array([8.220, -28.530, 0], dtype=np.float64), 16.333, top_ball_height, bot_ball_height)
    series_panel_array.link_panels(kc_s3_s4, "S3", "S4")
    kc_s4_s5 = ShortArmKinematicCoupling("S4_S5", np.array([2.125, -24.193, 0], dtype=np.float64), np.array([-9.825, -36.397, 0], dtype=np.float64), 14.749, bot_ball_height, arm_length, top_ball_height)
    series_panel_array.link_panels(kc_s4_s5, "S4", "S5")

    series_panel_array.visualize_array_3d()
    plt.show()

    # Run the Monte Carlo simulation
    series_panel_array.run_monte_carlo_simulation(kc_id_list, num_iterations, save_location, misalignment_std_dev)
    # The simulation will save the results in the specified directory

def get_baseline_series_panel_array():
    series_panel_array = LinkedPanelArray("P", np.array([-7.990, 0.000, 0.000], dtype=np.float64))

    panel_s1 = Panel("S1", np.array([2.665, 2.061, 0], dtype=np.float64))
    series_panel_array.add_panel(panel_s1)
    panel_s2 = Panel("S2", np.array([6.522, -0.245, 0], dtype=np.float64))
    series_panel_array.add_panel(panel_s2)
    panel_s3 = Panel("S3", np.array([9.069, -9.282, 0], dtype=np.float64))
    series_panel_array.add_panel(panel_s3)
    panel_s4 = Panel("S4", np.array([7.277, -20.091, 0], dtype=np.float64))
    series_panel_array.add_panel(panel_s4)
    panel_s5 = Panel("S5", np.array([-3.896, -30.431, 0], dtype=np.float64))
    series_panel_array.add_panel(panel_s5)
    return series_panel_array


if __name__ == "__main__":

    # Set parameters for the simulation
    kc_id_list = ["P_S1", "S1_S2", "S2_S3", "S3_S4", "S4_S5"]
    num_iterations = 10000
    master_save_location = r"C:\Users\thetk\OneDrive\Documents\GitHub\NASA-Model-T-KC-Simulation\src\monte_carlo_sims"
    misalignment_std_dev = 0.025  # Standard deviation for misalignment in cm

    long_arm_monte_carlo(get_baseline_series_panel_array(), kc_id_list, num_iterations, master_save_location, misalignment_std_dev)

    short_arm_monte_carlo(get_baseline_series_panel_array(), kc_id_list, num_iterations, master_save_location, misalignment_std_dev)


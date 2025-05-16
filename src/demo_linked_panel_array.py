"""
This module demonstrates the creation and visualization of various
LinkedPanelArray objects and their methods from the `panel_kc_errors_v1` module.

Its process was designed and implemented by Trevor K. Carter as an
employee of the CMR at BYU for a lab contract with NASA.
It was last modified on 05/--/2025.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import networkx as nx

from panel_kc_errors_v1 import KinematicCoupling, MaxwellKinematicCoupling, ShortArmKinematicCoupling, SideArm3BallKinematicCoupling
from panel_kc_errors_v1 import Panel, LinkedPanelArray


def demo_linear_panel_array():
    linked_panels = LinkedPanelArray("ground", np.array([-1, 0, 0], dtype=np.float64))

    panel_1 = Panel("1", np.array([1, 0, 0], dtype=np.float64))
    linked_panels.add_panel(panel_1)
    panel_2 = Panel("2", np.array([3, 0, 0], dtype=np.float64))
    linked_panels.add_panel(panel_2)
    panel_3 = Panel("3", np.array([5, 0, 0], dtype=np.float64))
    linked_panels.add_panel(panel_3)

    kc_g_1 = MaxwellKinematicCoupling("g_1", np.array([0, 0, 0], dtype=np.float64), np.array([1, 0, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_g_1, "ground", "1")
    kc_1_2 = MaxwellKinematicCoupling("1_2", np.array([2, 0, 0], dtype=np.float64), np.array([3, 0, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_1_2, "1", "2")
    kc_2_3 = MaxwellKinematicCoupling("2_3", np.array([4, 0, 0], dtype=np.float64), np.array([5, 0, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_2_3, "2", "3")

    #linked_panels.visualize_panel_graph()
    #plt.show()

    linked_panels.visualize_array_3d()
    plt.show()

    root_node, linked_panels_sub = linked_panels.get_kc_path(["g_1", "1_2"])
    linked_panels.visualize_array_3d(linked_panels_sub)
    plt.show()

    default_pose = linked_panels.get_default_poses(root_node, linked_panels_sub)
    print(default_pose)

    misalignment_1 = {'g_1': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64),
                     '1_2': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)}
    pose1_df = linked_panels.get_misalignment_propagated_world_poses(root_node, linked_panels_sub, misalignment_1)
    print(pose1_df[["panel_cm_final_pos", "net_rotation"]])
    pose1_df_optical = linked_panels.world_to_optical_pose(root_node, linked_panels_sub, pose1_df)
    print(pose1_df_optical)
    linked_panels.visualize_posed_array_3d(root_node, linked_panels_sub, pose1_df)
    plt.show()
    
    misalignments_2 = {'g_1': np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1], dtype=np.float64),
                     '1_2': np.array([0.0, 0.0, 0.0, 0.0, -0.1, -0.1], dtype=np.float64)}
    pose2_df = linked_panels.get_misalignment_propagated_world_poses(root_node, linked_panels_sub, misalignments_2)
    print(pose2_df[["panel_cm_final_pos", "net_rotation"]])
    pose2_df_optical = linked_panels.world_to_optical_pose(root_node, linked_panels_sub, pose2_df)
    print(pose2_df_optical)
    linked_panels.visualize_posed_array_3d(root_node, linked_panels_sub, pose2_df)
    plt.show()


def demo_zigzag_panel_array():
    linked_panels = LinkedPanelArray("ground", np.array([-1, 0, 0], dtype=np.float64))

    panel_1 = Panel("1", np.array([1, 0.5, 0], dtype=np.float64))
    linked_panels.add_panel(panel_1)
    panel_2 = Panel("2", np.array([3, -0.5, 0], dtype=np.float64))
    linked_panels.add_panel(panel_2)
    panel_3 = Panel("3", np.array([5, 0.5, 0], dtype=np.float64))
    linked_panels.add_panel(panel_3)

    kc_g_1 = MaxwellKinematicCoupling("g_1", np.array([0, 0.25, 0], dtype=np.float64), np.array([1, 0.5, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_g_1, "ground", "1")
    kc_1_2 = MaxwellKinematicCoupling("1_2", np.array([2, 0, 0], dtype=np.float64), np.array([3, -0.5, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_1_2, "1", "2")
    kc_2_3 = MaxwellKinematicCoupling("2_3", np.array([4, 0, 0], dtype=np.float64), np.array([5, 0.5, 0], dtype=np.float64), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_2_3, "2", "3")

    #linked_panels.visualize_panel_graph()
    #plt.show()

    linked_panels.visualize_array_3d()
    plt.show()


def demo_modelt_series_array():
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

    #series_panel_array.visualize_panel_graph()
    #plt.show()

    series_panel_array.visualize_array_3d()
    plt.show()

    root_node, linked_panels_sub = series_panel_array.get_kc_path(['P_S1', 'S1_S2', 'S2_S3', 'S3_S4', 'S4_S5'])
    series_panel_array.visualize_array_3d(linked_panels_sub)
    plt.show()

    default_pose = series_panel_array.get_default_poses(root_node, linked_panels_sub)
    print(default_pose)

    misalignment_1 = {'P_S1': np.array([0, 0, 0, 0, 0, 0.1], dtype=np.float64),
                     'S1_S2': np.array([0, 0, 0, 0, 0, 0.1], dtype=np.float64),
                     'S2_S3': np.array([0, 0, 0, 0, 0, 0.1], dtype=np.float64),
                     'S3_S4': np.array([0, 0, 0, 0, 0, 0.1], dtype=np.float64),
                     'S4_S5': np.array([0, 0, 0, 0, 0, 0.1], dtype=np.float64)}
    pose1_df = series_panel_array.get_misalignment_propagated_world_poses(root_node, linked_panels_sub, misalignment_1)
    print(pose1_df[["panel_cm_final_pos", "net_rotation"]])
    pose1_df_optical = series_panel_array.world_to_optical_pose(root_node, linked_panels_sub, pose1_df)
    print(pose1_df_optical)
    series_panel_array.visualize_posed_array_3d(root_node, linked_panels_sub, pose1_df)
    plt.show()

def demo_modelt_montecarloset_array():
    panel_collection_array = LinkedPanelArray("P", np.array([-7.990, 0.000, 0.000], dtype=np.float64))

    panel_s1 = Panel("S1", np.array([2.665, 2.061, 0], dtype=np.float64))
    panel_collection_array.add_panel(panel_s1)
    panel_s2 = Panel("S2", np.array([6.522, -0.245, 0], dtype=np.float64))
    panel_collection_array.add_panel(panel_s2)
    panel_s3 = Panel("S3", np.array([9.069, -9.282, 0], dtype=np.float64))
    panel_collection_array.add_panel(panel_s3)
    panel_s4 = Panel("S4", np.array([7.277, -20.091, 0], dtype=np.float64))
    panel_collection_array.add_panel(panel_s4)
    panel_s5 = Panel("S5", np.array([-3.896, -30.431, 0], dtype=np.float64))
    panel_collection_array.add_panel(panel_s5)

    panel_s1m = Panel("S1-", np.array([-6.778, 10.935, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s1m)
    panel_s2m = Panel("S2-", np.array([-3.392, 13.891, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s2m)
    panel_s3m = Panel("S3-", np.array([5.989, 13.521, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s3m)
    panel_s4m = Panel("S4-", np.array([15.716, 8.476, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s4m)
    panel_s5m = Panel("S5-", np.array([22.096, -5.345, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s5m)

    panel_s1p = Panel("S1+", np.array([-2.857, -9.661, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s1p)
    panel_s2p = Panel("S2+", np.array([-3.859, -14.043, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s2p)
    panel_s3p = Panel("S3+", np.array([-11.666, -19.257, 0.000], dtype=np.float64))
    panel_collection_array.add_panel(panel_s3p)

    arm_length = 5.0
    kc_p_s1m = SideArm3BallKinematicCoupling("P_S1-", np.array([-5.641, 7.764, 0], dtype=np.float64), np.array([-4.443, 11.452, 0], dtype=np.float64), 11.649, arm_length, 1.616)
    panel_collection_array.link_panels(kc_p_s1m, "P", "S1-")
    kc_s1m_s2m = SideArm3BallKinematicCoupling("S1-_S2-", np.array([-4.539, 11.654, 0], dtype=np.float64), np.array([-0.785, 14.625, 0], dtype=np.float64), 14.431, arm_length, 1.034)
    panel_collection_array.link_panels(kc_s1m_s2m, "S1-", "S2-")
    kc_s2m_s3m = SideArm3BallKinematicCoupling("S2-_S3-", np.array([-0.522, 12.089, 0], dtype=np.float64), np.array([11.805, 13.120, 0], dtype=np.float64), 12.474, arm_length, 1.619)
    panel_collection_array.link_panels(kc_s2m_s3m, "S2-", "S3-")
    kc_s3m_s4m = SideArm3BallKinematicCoupling("S3-_S4-", np.array([11.999, 13.370, 0], dtype=np.float64), np.array([24.033, 6.766, 0], dtype=np.float64), 16.333, arm_length, 1.034)
    panel_collection_array.link_panels(kc_s3m_s4m, "S3-", "S4-")
    kc_s4m_s5m = SideArm3BallKinematicCoupling("S4-_S5-", np.array([18.027, 2.319, 0], dtype=np.float64), np.array([25.938, -12.828, 0], dtype=np.float64), 14.749, arm_length, 1.606, 0, np.pi/2, "right")
    panel_collection_array.link_panels(kc_s4m_s5m, "S4-", "S5-")

    kc_p_s1 = SideArm3BallKinematicCoupling("P_S1", np.array([0.000, 0.000, 0], dtype=np.float64), np.array([3.878, 0.000, 0], dtype=np.float64), 11.649, arm_length, 1.616)
    panel_collection_array.link_panels(kc_p_s1, "P", "S1")
    kc_s1_s2 = SideArm3BallKinematicCoupling("S1_S2", np.array([4.040, 0.155, 0], dtype=np.float64), np.array([8.026, -2.498, 0], dtype=np.float64), 14.431, arm_length, 1.034)
    panel_collection_array.link_panels(kc_s1_s2, "S1", "S2")
    kc_s2_s3 = SideArm3BallKinematicCoupling("S2_S3", np.array([5.695, -3.532, 0], dtype=np.float64), np.array([10.485, -14.937, 0], dtype=np.float64), 12.474, arm_length, 1.619)
    panel_collection_array.link_panels(kc_s2_s3, "S2", "S3")
    kc_s3_s5m = SideArm3BallKinematicCoupling("S3_S5-", np.array([15.178, -8.807, 0], dtype=np.float64), np.array([29.165, -2.119, 0], dtype=np.float64), 17.094, arm_length, 1.536, 0, np.pi/2, "right")
    panel_collection_array.link_panels(kc_s3_s5m, "S3", "S5-")
    kc_s3_s4 = SideArm3BallKinematicCoupling("S3_S4", np.array([10.782, -15.782, 0], dtype=np.float64), np.array([8.220, -28.530, 0], dtype=np.float64), 16.333, arm_length, 1.034)
    panel_collection_array.link_panels(kc_s3_s4, "S3", "S4")
    kc_s4_s5 = SideArm3BallKinematicCoupling("S4_S5", np.array([2.125, -24.193, 0], dtype=np.float64), np.array([-9.825, -36.397, 0], dtype=np.float64), 14.749, arm_length, 1.606, 0, np.pi/2, "right")
    panel_collection_array.link_panels(kc_s4_s5, "S4", "S5")

    kc_p_s1p = SideArm3BallKinematicCoupling("P_S1+", np.array([-5.641, -7.764, 0], dtype=np.float64), np.array([-4.442, -11.452, 0], dtype=np.float64), 11.649, arm_length, 1.616)
    panel_collection_array.link_panels(kc_p_s1p, "P", "S1+")
    kc_s1p_s2p = SideArm3BallKinematicCoupling("S1+_S2+", np.array([-4.246, -11.559, 0], dtype=np.float64), np.array([-5.537, -16.169, 0], dtype=np.float64), 14.431, arm_length, 1.034)
    panel_collection_array.link_panels(kc_s1p_s2p, "S1+", "S2+")
    kc_s2p_s3p = SideArm3BallKinematicCoupling("S2+_S3+", np.array([-7.240, -14.272, 0], dtype=np.float64), np.array([-16.607, -22.352, 0], dtype=np.float64), 12.474, arm_length, 1.619)
    panel_collection_array.link_panels(kc_s2p_s3p, "S2+", "S3+")
    kc_s3p_s5 = SideArm3BallKinematicCoupling("S3+_S5", np.array([-9.326, -24.920, 0], dtype=np.float64), np.array([1.356, -36.157, 0], dtype=np.float64), 17.094, arm_length, 1.536, 0, np.pi/2, "right")
    panel_collection_array.link_panels(kc_s3p_s5, "S3+", "S5")

    #panel_collection_array.visualize_panel_graph()
    #plt.show()

    panel_collection_array.visualize_array_3d(font_modifier=0.6)
    plt.show()



if __name__ == "__main__":
    #demo_linear_panel_array()
    #demo_zigzag_panel_array()
    demo_modelt_series_array()
    #demo_modelt_montecarloset_array()
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
    linked_panels = LinkedPanelArray("ground", np.array([-1, 0, 0], dtype=float))

    panel_1 = Panel("1", np.array([1, 0, 0], dtype=float))
    linked_panels.add_panel(panel_1)
    panel_2 = Panel("2", np.array([3, 0, 0], dtype=float))
    linked_panels.add_panel(panel_2)
    panel_3 = Panel("3", np.array([5, 0, 0], dtype=float))
    linked_panels.add_panel(panel_3)

    kc_g_1 = MaxwellKinematicCoupling("g_1", np.array([0, 0, 0], dtype=float), np.array([1, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_g_1, "ground", "1")
    kc_1_2 = MaxwellKinematicCoupling("1_2", np.array([2, 0, 0], dtype=float), np.array([3, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_1_2, "1", "2")
    kc_2_3 = MaxwellKinematicCoupling("2_3", np.array([4, 0, 0], dtype=float), np.array([5, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_2_3, "2", "3")

    #linked_panels.visualize_panel_graph()
    #plt.show()

    linked_panels.visualize_panel_3d()
    plt.show()

def demo_zigzag_panel_array():
    linked_panels = LinkedPanelArray("ground", np.array([-1, 0, 0], dtype=float))

    panel_1 = Panel("1", np.array([1, 0.5, 0], dtype=float))
    linked_panels.add_panel(panel_1)
    panel_2 = Panel("2", np.array([3, -0.5, 0], dtype=float))
    linked_panels.add_panel(panel_2)
    panel_3 = Panel("3", np.array([5, 0.5, 0], dtype=float))
    linked_panels.add_panel(panel_3)

    kc_g_1 = MaxwellKinematicCoupling("g_1", np.array([0, 0.25, 0], dtype=float), np.array([1, 0.5, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_g_1, "ground", "1")
    kc_1_2 = MaxwellKinematicCoupling("1_2", np.array([2, 0, 0], dtype=float), np.array([3, -0.5, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_1_2, "1", "2")
    kc_2_3 = MaxwellKinematicCoupling("2_3", np.array([4, 0, 0], dtype=float), np.array([5, 0.5, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_2_3, "2", "3")

    #linked_panels.visualize_panel_graph()
    #plt.show()

    linked_panels.visualize_panel_3d()
    plt.show()


def demo_modelt_petal_array():
    linked_panels = LinkedPanelArray("pent", np.array([-1, 0, 0], dtype=float))

    panel_1 = Panel("a1", np.array([1, 0, 0], dtype=float))
    linked_panels.add_panel(panel_1)
    panel_2 = Panel("2", np.array([3, 0, 0], dtype=float))
    linked_panels.add_panel(panel_2)
    panel_3 = Panel("3", np.array([5, 0, 0], dtype=float))
    linked_panels.add_panel(panel_3)

    kc_g_1 = MaxwellKinematicCoupling("g_1", np.array([0, 0, 0], dtype=float), np.array([1, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_g_1, "pent", "1")
    kc_1_2 = MaxwellKinematicCoupling("1_2", np.array([2, 0, 0], dtype=float), np.array([3, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_1_2, "1", "2")
    kc_2_3 = MaxwellKinematicCoupling("2_3", np.array([4, 0, 0], dtype=float), np.array([5, 0, 0], dtype=float), 1, 0.5, -0.5)
    linked_panels.link_panels(kc_2_3, "2", "3")

    #linked_panels.visualize_panel_graph()
    #plt.show()

    linked_panels.visualize_panel_3d()
    plt.show()



if __name__ == "__main__":
    demo_linear_panel_array()
    demo_zigzag_panel_array()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import networkx as nx

from chatgpt_demo import estimate_pose, plot_pose_estimation




class KinematicCoupling():
    """
    A class to represent a kinematic coupling. It solves for the misalignment poses of the panel based on all combinations of misalignment values.
    """
    def __init__(self, kc_id: str, kc_origin: np.ndarray[3], kc_origin_norm: np.ndarray[3], kc_contact_points: np.ndarray[6, 3], kc_contact_norms: np.ndarray[6, 3]):
        """
        Add information about a kinematic coupling for a panel.
        :param kc_origin: origin of the kinematic coupling in the array coordinate frame
        :param kc_origin_norm: normal of the kinematic coupling in the array coordinate frame
        :param kc_contact_points: 6 contact point coordinates of the kinematic coupling in the kc coordinate frame*
        :param kc_contact_norms: 6 unit normals corresponding to planar contact surfaces in the kinematic coupling, in the kc coordinate frame*

        *Note: the contact points and normals are in the kinematic coupling local coordinate frame, which is defined as follows:
        - The coordinate frame origin is at the kinematic coupling origin
        - The x axis is the direction of the kc_origin_norm
        - The y axis is the direction of the length of the kinematic coupling
        - The z axis is orthogonal to the x and y axes
        """
        self.kc_id = kc_id
        self.kc_origin = kc_origin
        self.kc_origin_norm = kc_origin_norm
        self.kc_contact_points = kc_contact_points
        self.kc_contact_norms = kc_contact_norms

    def solve_pose(self, misalignment: np.ndarray[6]):
        """
        Solve for the misalignment pose of the panel based on a single set of misalignment values. Return and do not save the result.
        :param misalignment: row of 6 misalignment values to be evaluated, each corresponding to translation of a kc contact plane along its normal
        """
        result = np.array(estimate_pose(self.kc_contact_points, self.kc_contact_norms, misalignment).x)
        return result
    
    def plot_solve_kc_pose(self, misalignments: np.ndarray[6], ax=None):
        """
        Solve then plot the kinematic coupling pose in 3D space. The x axis of the kc is shown as the graph's z axis.
        :param misalignments: row of 6 misalignment values to be evaluated, each corresponding to translation of a kc contact plane along its normal
        :param ax: axis to plot on (default is None, which creates a new figure)
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Solve kc_pose
        rot_trans_pose = self.solve_pose(misalignments)

        self.plot_kc_pose(rot_trans_pose, misalignments, ax=ax)

    def plot_kc_pose(self, rot_trans_pose, misalignments, ax=None):
        """
        Plot the kinematic coupling pose in 3D space. The x axis of the kc is shown as the graph's z axis.
        :param rot_trans_pose: rotation and translation vectors of the kinematic coupling pose
        :param misalignments: row of 6 misalignment values that produced the pose above, each corresponding to translation of a kc contact plane along its normal
        :param ax: axis to plot on (default is None, which creates a new figure)
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Define the transformation matrix to swap axes: x -> z, z -> y, y -> x
        transformation_matrix = np.array([
            [0, 1, 0],  # y -> x
            [0, 0, 1],  # z -> y
            [1, 0, 0]   # x -> z
        ])

        # Transform body_points and contact_norms
        body_points = (transformation_matrix @ self.kc_contact_points.T).T
        contact_norms = (transformation_matrix @ self.kc_contact_norms.T).T

        transformed_rotation_vector = transformation_matrix @ rot_trans_pose[:3]
        transformed_translation_vector = transformation_matrix @ rot_trans_pose[3:]
        kc_pose = np.concatenate([transformed_rotation_vector, transformed_translation_vector])

        plot_pose_estimation(body_points, contact_norms, misalignments, kc_pose, ax=ax)
    

    
class MaxwellKinematicCoupling(KinematicCoupling):
    """
    A class to represent a short arm kinematic coupling. It is a pre-determined kinematic coupling configuration that can prepared with fewer parameters than defining all contact points and contact normals.
    """
    def __init__(self, kc_id, kc_origin: np.ndarray[3], kc_origin_norm: np.ndarray[3], kc_interface_length: float, kc_peak_height_offset: float, kc_base_height_offset: float, kc_peak_pos: float = 0, kc_groove_angle: float = np.pi/2, kc_groove_orientation: str = "centroid"):
        """
        Add information about a kinematic coupling for a panel.
        :param kc_origin: origin of the kinematic coupling in the array coordinate frame
        :param kc_origin_norm: normal of the kinematic coupling in the array coordinate frame
        :param kc_interface_length: length of the kinematic coupling interface
        :param kc_interface_height_offset: height offset of the balls in grooves on the kinematic coupling interface surface
        :param kc_peak_height_offset: height offset of the ball making the peak of the Maxwell kinematic coupling
        :param kc_base_height_offset: height offset of the balls making the base of the Maxwell kinematic coupling
        :param kc_peak_pos: position along the y axis where the peak is located (default is 0, which is the center of the kinematic coupling interface)
        :param kc_groove_angle: angle of the grooves in the kinematic couplings (default is pi/2)
        :param kc_groove_orientation: orientation of the grooves in the kinematic couplings (default is "centroid", can also be "center" or "orthogonal")

        Observe the Parent class for more information about the coordinate frame of the kinematic coupling attributes.
        """
        # Define the contact points and normals in the kc coordinate frame
        contact_points = np.array([
            [0, -kc_interface_length/2, kc_base_height_offset],
            [0, -kc_interface_length/2, kc_base_height_offset],
            [0, kc_interface_length/2, kc_base_height_offset],
            [0, kc_interface_length/2, kc_base_height_offset],
            [0, kc_peak_pos, kc_peak_height_offset],
            [0, kc_peak_pos, kc_peak_height_offset]
        ], dtype=float)

        x_phi = kc_groove_angle/2
        if kc_groove_orientation == "centroid":
            centroid = np.array([0, kc_peak_pos/3, (kc_peak_height_offset + 2*kc_base_height_offset)/3])
            vec_to_base_right = contact_points[1] - centroid
            vec_to_base_left = contact_points[3] -  centroid
            vec_to_peak = contact_points[5] - centroid
        elif kc_groove_orientation == "center":
            vec_to_base_right = contact_points[1]
            vec_to_base_left = contact_points[3]
            vec_to_peak = contact_points[5]
        elif kc_groove_orientation == "orthogonal":
            vec_to_base_right = np.array([0, -1, 0])
            vec_to_base_left = np.array([0, 1, 0])
            vec_to_peak = np.array([0, 0, 1])

        rb_1, rb_2 = self._get_maxwell_norm_vectors(x_phi, vec_to_base_right)
        lb_1, lb_2 = self._get_maxwell_norm_vectors(x_phi, vec_to_base_left)
        p_1, p_2 = self._get_maxwell_norm_vectors(x_phi, vec_to_peak)
        
        contact_norms = np.array([
            rb_1,
            rb_2,
            lb_1,
            lb_2,
            p_1,
            p_2
        ], dtype=float)

        super().__init__(kc_id, kc_origin, kc_origin_norm, contact_points, contact_norms)

    def _get_maxwell_norm_vectors(self, phi_x, v_cent_to_groove):
        """
        This function is used to calculate the directions of the surface normals of the grooves in the Maxwell kinematic coupling.

        Given:
        - theta: angle in radians
        - v: 3-vector (array-like) orthogonal to the x-axis (i.e. v[0]=0), not necessarily unit
        Returns:
        (u1, u2): the two unit 3-vectors that
            • lie in the plane perpendicular to v,
            • make angle theta with the +x axis,
            • and are orthogonal to v.
        """
        v_cent_to_groove = np.asarray(v_cent_to_groove, dtype=float)
        # ensure it's orthogonal to x-axis:
        if abs(v_cent_to_groove[0]) > 1e-8:
            raise ValueError("Input vector must have zero x-component")
        # normalize v in its own (yz) subspace
        vy, vz = v_cent_to_groove[1], v_cent_to_groove[2]
        mag_yz = np.hypot(vy, vz)
        if mag_yz < 1e-8:
            raise ValueError("Input vector must not be zero in its yz-components")
        vy /= mag_yz
        vz /= mag_yz

        c, s = np.cos(phi_x), np.sin(phi_x)

        # choose the unit-vector in the yz-plane orthogonal to v:
        #    w = (-vz, vy)
        # then the two solutions are u = [c, ± s·w]
        u1 = np.array([c, -vz * s, vy * s])
        u2 = np.array([c, vz * s, -vy * s])

        return u1, u2
    
class ShortArmKinematicCoupling(KinematicCoupling):
    """
    A class to represent a short arm kinematic coupling. It is a pre-determined kinematic coupling configuration that can prepared with fewer parameters than defining all contact points and contact normals.
    """
    def __init__(self, kc_id, kc_origin: np.ndarray[3], kc_origin_norm: np.ndarray[3], kc_interface_length: float, kc_interface_height_offset: float, kc_arm_length: float, kc_arm_height_offset: float, kc_arm_pos: float = 0, kc_groove_angle: float = np.pi/2):
        """
        Add information about a kinematic coupling for a panel.
        :param kc_origin: origin of the kinematic coupling in the array coordinate frame
        :param kc_origin_norm: normal of the kinematic coupling in the array coordinate frame
        :param kc_interface_length: length of the kinematic coupling interface
        :param kc_interface_height_offset: height offset of the balls in grooves on the kinematic coupling interface surface
        :param kc_arm_length: length of the kinematic coupling arm
        :param kc_arm_height_offset: height offset of the ball in a groove on the kinematic coupling arm surface
        :param kc_arm_pos: position along the y axis where the arm is located (default is 0, which is the center of the kinematic coupling interface)
        :param kc_groove_angle: angle of the groove on the kinematic coupling arm (default is pi/2)

        Observe the Parent class for more information about the coordinate frame of the kinematic coupling attributes.
        """
        # Define the contact points and normals in the kc coordinate frame
        contact_points = np.array([
            [0, -kc_interface_length/2, kc_interface_height_offset],
            [0, -kc_interface_length/2, kc_interface_height_offset],
            [0, kc_interface_length/2, kc_interface_height_offset],
            [0, kc_interface_length/2, kc_interface_height_offset],
            [kc_arm_length, kc_arm_pos, kc_arm_height_offset],
            [kc_arm_length, kc_arm_pos, kc_arm_height_offset]
        ], dtype=float)

        a = np.cos(kc_groove_angle/2)
        b = np.sin(kc_groove_angle/2)
        contact_norms = np.array([
            [a, 0, b],
            [a, 0, -b],
            [a, 0, b],
            [a, 0, -b],
            [0, b, -a],
            [0, -b, -a]
        ], dtype=float)

        super().__init__(kc_id, kc_origin, kc_origin_norm, contact_points, contact_norms)

    
class SideArm3BallKinematicCoupling(KinematicCoupling):
    """
    A class to represent a left arm kinematic coupling. It is a pre-determined kinematic coupling configuration that can prepared with fewer parameters than defining all contact points and contact normals.
    """
    def __init__(self, kc_id, kc_origin: np.ndarray[3], kc_origin_norm: np.ndarray[3], kc_interface_length: float, kc_arm_length: float, kc_interface_arm_angle: float = np.pi/2, kc_arm_ball_height_offset: float = 0, kc_groove_angle: float = np.pi/2, kc_arm_side: str = "left"):
        """
        Add information about a kinematic coupling for a panel.
        :param kc_origin: origin of the kinematic coupling in the array coordinate frame
        :param kc_origin_norm: normal of the kinematic coupling in the array coordinate frame
        :param kc_interface_length: length of the kinematic coupling interface
        :param kc_arm_length: length of the kinematic coupling arm
        :param kc_interface_arm_angle: angle of the kinematic coupling arm with respect to the interface (default is pi/2)
        :param kc_arm_ball_height_offset: height offset of the ball center in the kinematic coupling arm surface (default is 0)
        :param kc_groove_angle: angle of the groove on the kinematic coupling arm (default is pi/2)
        :param kc_arm_side: side of the kinematic coupling arm (default is "left", can also be "right")

        Observe the Parent class for more information about the coordinate frame of the kinematic coupling attributes.
        """
        # Define the contact points and normals in the kc coordinate frame
        if kc_arm_side == "left":
            arm_pos = kc_interface_length/2
        elif kc_arm_side == "right":
            arm_pos = -kc_interface_length/2
        else:
            raise ValueError("Invalid arm side. Must be 'left' or 'right'.")
        contact_points = np.array([
            [0, -arm_pos, 0],
            [0, -arm_pos, 0],
            [0, arm_pos, 0],
            [0, arm_pos, 0],
            [0, arm_pos, 0],
            [kc_arm_length * np.sin(kc_interface_arm_angle), arm_pos + kc_arm_length * np.cos(kc_interface_arm_angle) * (-1 if kc_arm_side == "left" else 1), kc_arm_ball_height_offset],
        ], dtype=float)

        a = np.cos(kc_groove_angle/2)
        b = np.sin(kc_groove_angle/2)
        contact_norms = np.array([
            [a, 0, b],
            [a, 0, -b],
            [a, b, 0],
            [a, -b, 0],
            [0, 0, -1],
            [0, 0, -1]
        ], dtype=float)

        super().__init__(kc_id, kc_origin, kc_origin_norm, contact_points, contact_norms)

    


class Panel():
    """
    A class to represent a panel object. 
    """
    def __init__(self, panel_id, panel_cm):
        self.panel_id = panel_id
        self.panel_cm = panel_cm
        self.kinematic_couplings = dict()

    def add_panel_kc(self, kc_object: KinematicCoupling):
        """
        Add a kinematic coupling to the panel.
        :param kc_object: KinematicCoupling object
        """
        # Check if the kinematic coupling id already exists
        if kc_object.kc_id in self.kinematic_couplings:
            raise ValueError(f"Kinematic coupling with id {kc_object.kc_id} already exists.")
        # Add the kinematic coupling object to the panel
        self.kinematic_couplings[kc_object.kc_id] = kc_object

    def get_data_post_kc_misalignment(self, kc_id: str, misalignments: np.ndarray[6], external_rotation: R = None, external_translation: np.ndarray[3] = None, linked_kc_origins: list[np.ndarray[3]] = None) -> tuple:
        """
        After applying each evaluated misalignment for some kinematic coupling, get more information about the panel based on the misalignment pose.
        :param kc_id: id of the kinematic coupling
        :param misalignments: array of misalignment values, each corresponding to translation of a kc contact plane along its normal
        :param external_rotation: rotation of the panel in the array coordinate frame derived from its parent (omitted for the first panel)
        :param external_translation: translation of the panel in the array coordinate frame derived from its parent (omitted for the first panel)
        :param linked_kc_origins: array of origins of the kinematic couplings linked to the panel (omitted for panels with no linked kinematic couplings)
        :return: tuple of the following:
            - resulting_kc_origin: origin of the kinematic coupling in the array coordinate frame after applying the misalignment pose
            - resulting_cm: center of mass of the panel in the array coordinate frame after applying the misalignment pose
            - resulting_kc_norm: normal of the kinematic coupling in the array coordinate frame after applying the misalignment pose
            - net_rotation: rotation of the panel in the array coordinate frame after applying the misalignment pose
            - net_origin_translation: translation of the origin of the panel in the array coordinate frame after applying the misalignment pose
            - resulting_linked_kc_shifts: shifts of the linked kinematic couplings' origins in the array coordinate frame after applying the misalignment pose (omitted for panels with no linked kinematic couplings)
        """
        # Find the kinematic coupling object with the given id
        if kc_id not in self.kinematic_couplings:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found in set of kinematic couplings.")
        kc_object = self.kinematic_couplings[kc_id]
        # Note that the kc_pose is in panel coordinates and must be transformed to array coordinates
        kc_pose = kc_object.solve_pose(misalignments)
        # Note that all of the following are in array coordinates
        kc_origin = kc_object.kc_origin
        kc_to_norm = kc_object.kc_origin_norm - kc_origin
        kc_to_cm = self.panel_cm - kc_origin
        if linked_kc_origins is not None:
            # Get vectors from kc_origin to the linked kinematic couplings' origns
            origin_kc_to_linked_kcs = np.array(linked_kc_origins, ndmin=2, copy=True) - kc_origin
        else:
            origin_kc_to_linked_kcs = None

        # Convert kc_origin_norm to a unit vector
        kc_x_axis = kc_to_norm / np.linalg.norm(kc_to_norm)

        # Define the panel coordinate system axes in array coordinates
        kc_z_axis = np.array([0, 0, 1])  # Assuming array z-axis is aligned with panel z-axis
        kc_y_axis = np.cross(kc_z_axis, kc_x_axis)
        kc_y_axis /= np.linalg.norm(kc_y_axis)
        kc_z_axis = np.cross(kc_x_axis, kc_y_axis)

        # Construct the rotation matrix from panel coordinates to array coordinates
        panel_to_array_transformation = np.column_stack((kc_x_axis, kc_y_axis, kc_z_axis))
        
        rotation_panel_coords = kc_pose[:3]  # [:3] rotation vector in panel coordinates
        translation_panel_coords = kc_pose[3:]  # [3:] translation vector in panel coordinates

        # Transform all misalignment rotations and translations into array coordinates
        rotation_array_coords = rotation_panel_coords @ panel_to_array_transformation.T
        local_rotation = R.from_rotvec(rotation_array_coords)
        net_rotation = (external_rotation * local_rotation) if external_rotation else local_rotation
        array_translation = translation_panel_coords @ panel_to_array_transformation.T
        if external_rotation is not None:
            array_translation = external_rotation.apply(array_translation)

        # Apply the rotations and translations
        resulting_kc_origin = kc_origin + array_translation
        if external_translation is not None:
            resulting_kc_origin += external_translation
        resulting_cm = net_rotation.apply(kc_to_cm) + resulting_kc_origin
        resulting_kc_norm = net_rotation.apply(kc_to_norm) + resulting_kc_origin

        if origin_kc_to_linked_kcs is not None:
            resulting_linked_kc_shifts = net_rotation.apply(origin_kc_to_linked_kcs) + array_translation - origin_kc_to_linked_kcs
            return resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation, resulting_linked_kc_shifts
        else:
            return resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation

class GroundPanel():
    """
    A class to represent a ground panel object. This is a special type of panel that is not linked to any other panels.
    """
    def __init__(self, ground_id, ground_cm):
        self.panel_id = ground_id
        self.panel_cm = ground_cm


class LinkedPanelArray():
    """
    V0 attempted to simultaneously solve for all poses based on all combinations of misalignment values. This became difficult to manage.
    V1 will solve for a single pose based on a single set of misalignment values for all kinematic couplings in the array.

    Working Process:
    1) Add panels to the array
    2) Add kinematic couplings to link pairs of panels
    3) Determine the set of misalignment values that will be investigated across all kinematic couplings
    4) Evaluate poses individually for all kinematic couplings based on the misalignment values
    """
    def __init__(self, ground_id, ground_cm=None) -> None:
        # Initialize a directed graph to store panels and kinematic couplings
        self.graph = nx.DiGraph()
        self.kc_id_to_nodes = {}  # Map kc_id to (parent_panel_id, child_panel_id)

        # Add the ground panel as the root node
        self.graph.add_node(ground_id, obj=GroundPanel(ground_id, ground_cm), type="ground")

    def add_panel(self, panel: Panel):
        if panel.panel_id in self.graph.nodes:
            raise ValueError(f"Panel with id {panel.panel_id} already exists.")
        # Add the panel as a node in the graph
        self.graph.add_node(panel.panel_id, obj=panel, type="panel")

    def link_panels(self, kc: KinematicCoupling, child_panel_id: str, parent_panel_id: str):
        """
        Link two panels together based on a kinematic coupling.
        :param kc: KinematicCoupling object linking the panels
        :param child_panel_id: id of the child panel, which receives the kc
        :param parent_panel_id: id of the parent panel, to which the child panel is linked by its kc
        """
        # Ensure both panels exist in the graph
        if child_panel_id not in self.graph.nodes or parent_panel_id not in self.graph.nodes:
            raise ValueError(f"Panel with id {child_panel_id} or {parent_panel_id} not found.")
        if kc.kc_id in self.kc_id_to_nodes:
            raise ValueError(f"Kinematic coupling with id {kc.kc_id} already exists.")

        # Add the kinematic coupling as an edge in the graph
        self.graph.add_edge(parent_panel_id, child_panel_id, obj=kc, type="kc", kc_id=kc.kc_id)

        # Map kc_id to the edge
        self.kc_id_to_nodes[kc.kc_id] = (parent_panel_id, child_panel_id)

        # Add the kinematic coupling to the child panel
        child_panel = self.graph.nodes[child_panel_id]["obj"]
        child_panel.add_panel_kc(kc)

    def get_kc_path(self, kc_ids: list[str]) -> tuple[str, nx.DiGraph]:
        """
        Propagate the misalignment poses for multiple kinematic couplings based on all combinations of misalignment values.
        :param kc_ids: variable number of kinematic coupling ids to propagate
        :return: tuple of the following:
            - lowest_root_node_id: id of the lowest root node in the path
            - subgraph: subgraph containing the nodes and edges to propagate
        """
        for kc_id in kc_ids:
            # Retrieve the edge using kc_id
            if kc_id not in self.kc_id_to_nodes:
                raise ValueError(f"Kinematic coupling with id {kc_id} not found.")
            
        node_ids_in_path = set()
        for kc_id in kc_ids:
            parent_panel_id, child_panel_id = self.kc_id_to_nodes[kc_id]
            node_ids_in_path.add(parent_panel_id)
            node_ids_in_path.add(child_panel_id)
        subgraph = self.graph.subgraph(node_ids_in_path)

        # Ensure all edges form a continuous path
        if not nx.is_connected(subgraph.to_undirected()):
            raise ValueError("The provided kinematic couplings do not form a continuous path.")
        
        # Ensure that there are no cycles in the subgraph
        if not nx.is_directed_acyclic_graph(subgraph):
            raise ValueError("The provided kinematic couplings form a cycle. This is not allowed.")
        
        # Ensure that no two edges in the subgraph point to the same node
        for node_id in node_ids_in_path:
            if subgraph.in_degree(node_id) > 1:
                raise ValueError(f"Panel with id {node_id} has more than one incoming kinematic coupling, which currently has no implementation. Consider using a different set of kinematic couplings.")

        # Find the lowest root node
        root_candidates_ids = [node_id for node_id in node_ids_in_path if self.graph.in_degree(node_id) == 0]
        if len(root_candidates_ids) != 1:
            raise ValueError("There is no unique root node for the provided kinematic couplings. In other words, there is no clear" \
            "starting point to evaluate misalignment propagation from for the panels linked by the provided kinematic couplings." \
            "Consider using a different set of kinematic couplings to satisfy this requirement.")
        lowest_root_node_id = root_candidates_ids[0]

        return lowest_root_node_id, subgraph
    
    def get_misalignment_propagated_poses(self, root_id: str, subgraph: nx.classes.DiGraph, misalignment_dict: dict[str, np.ndarray[6]]) -> pd.DataFrame:
        """
        Propagate the misalignment poses for multiple kinematic couplings based on all combinations of misalignment values.
        :param root_id: id of the root node
        :param subgraph: subgraph containing the nodes and edges to propagate
        :param misalignments: array of misalignment values, each corresponding to translation of a kc contact plane along its normal
        :return: DataFrame containing the misalignments and the resulting poses of the panels and kinematic couplings in the subgraph
        """
        if len(misalignment_dict) != len(subgraph.edges):
            raise ValueError("The number of misalignment values must match the number of kinematic couplings in the subgraph.")
                # Define the multiindex for the columns
        column_tuples = [
            ("kc_misalignment", "1"), ("kc_misalignment", "2"), ("kc_misalignment", "3"),
            ("kc_misalignment", "4"), ("kc_misalignment", "5"), ("kc_misalignment", "6"),
            ("kc_origin_final_pos", "x"), ("kc_origin_final_pos", "y"), ("kc_origin_final_pos", "z"),
            ("kc_norm_final_pos", "x"), ("kc_norm_final_pos", "y"), ("kc_norm_final_pos", "z"),
            ("panel_cm_final_pos", "x"), ("panel_cm_final_pos", "y"), ("panel_cm_final_pos", "z")
        ]
        multiindex = pd.MultiIndex.from_tuples(column_tuples, names=["Category", "Subcategory"])

        # Initialize the DataFrame with the multiindex and the number of rows equal to the number of nodes in the subgraph
        result_table = pd.DataFrame(columns=multiindex, index=sorted(subgraph.nodes))

        def next_panel_misalignments(parent_panel_id, ext_rot, linked_kc_shifts=None):
            if linked_kc_shifts is None:
                linked_kc_shifts = [None] * len(subgraph.successors(parent_panel_id))
            for child_panel_id, child_panel_kc_shift in zip(sorted(subgraph.successors(parent_panel_id)), linked_kc_shifts):
                # Get the panel object
                child_panel = self.graph.nodes[child_panel_id]["obj"]
                # Get the kinematic coupling id
                kc_id = subgraph[parent_panel_id][child_panel_id]["kc_id"]
                kc_misalignments = misalignment_dict[kc_id]
                if len(subgraph.successors(child_panel_id)) > 0:
                    linked_kc_ids = [subgraph[child_panel_id][grandchild_panel]["kc_id"] for grandchild_panel in sorted(subgraph.successors(child_panel_id))]
                    linked_kc_origins = [
                        subgraph[child_panel_id][self.kc_id_to_nodes[linked_kc_id][1]]["obj"].kc_origin
                        for linked_kc_id in linked_kc_ids
                    ]
                    resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation, resulting_linked_kc_shifts = child_panel.get_data_post_kc_misalignment(kc_id, kc_misalignments, external_rotation=ext_rot, external_translation=child_panel_kc_shift, linked_kc_origins=linked_kc_origins)
                    # Fill out the result table with the data
                    result_table.loc[child_panel_id] = pd.Series({
                        ("kc_misalignment", "1"): kc_misalignments[0],
                        ("kc_misalignment", "2"): kc_misalignments[1],
                        ("kc_misalignment", "3"): kc_misalignments[2],
                        ("kc_misalignment", "4"): kc_misalignments[3],
                        ("kc_misalignment", "5"): kc_misalignments[4],
                        ("kc_misalignment", "6"): kc_misalignments[5],
                        ("kc_origin_final_pos", "x"): resulting_kc_origin[0],
                        ("kc_origin_final_pos", "y"): resulting_kc_origin[1],
                        ("kc_origin_final_pos", "z"): resulting_kc_origin[2],
                        ("kc_norm_final_pos", "x"): resulting_kc_norm[0],
                        ("kc_norm_final_pos", "y"): resulting_kc_norm[1],
                        ("kc_norm_final_pos", "z"): resulting_kc_norm[2],
                        ("panel_cm_final_pos", "x"): resulting_cm[0],
                        ("panel_cm_final_pos", "y"): resulting_cm[1],
                        ("panel_cm_final_pos", "z"): resulting_cm[2],
                    })
                    # If the child panel has successors, recursively call the function to process those successors and propagate the misalignment poses
                    next_panel_misalignments(child_panel_id, net_rotation, resulting_linked_kc_shifts)
                else:
                    # If the child panel has no successors, just get its pose data post misalignment
                    resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation = child_panel.get_data_post_kc_misalignment(kc_id, kc_misalignments, external_rotation=ext_rot, external_translation=child_panel_kc_shift)
                    # Fill out the result table with the data
                    result_table.loc[child_panel_id] = pd.Series({
                        ("kc_misalignment", "1"): kc_misalignments[0],
                        ("kc_misalignment", "2"): kc_misalignments[1],
                        ("kc_misalignment", "3"): kc_misalignments[2],
                        ("kc_misalignment", "4"): kc_misalignments[3],
                        ("kc_misalignment", "5"): kc_misalignments[4],
                        ("kc_misalignment", "6"): kc_misalignments[5],
                        ("kc_origin_final_pos", "x"): resulting_kc_origin[0],
                        ("kc_origin_final_pos", "y"): resulting_kc_origin[1],
                        ("kc_origin_final_pos", "z"): resulting_kc_origin[2],
                        ("kc_norm_final_pos", "x"): resulting_kc_norm[0],
                        ("kc_norm_final_pos", "y"): resulting_kc_norm[1],
                        ("kc_norm_final_pos", "z"): resulting_kc_norm[2],
                        ("panel_cm_final_pos", "x"): resulting_cm[0],
                        ("panel_cm_final_pos", "y"): resulting_cm[1],
                        ("panel_cm_final_pos", "z"): resulting_cm[2],
                    })
        
        # The root node is the first node in the subgraph. It has no misalignment values or kc
        result_table.loc[root_id] = pd.Series({
            ("kc_misalignment", "1"): np.nan,
            ("kc_misalignment", "2"): np.nan,
            ("kc_misalignment", "3"): np.nan,
            ("kc_misalignment", "4"): np.nan,
            ("kc_misalignment", "5"): np.nan,
            ("kc_misalignment", "6"): np.nan,
            ("kc_origin_final_pos", "x"): np.nan,
            ("kc_origin_final_pos", "y"): np.nan,
            ("kc_origin_final_pos", "z"): np.nan,
            ("kc_norm_final_pos", "x"): np.nan,
            ("kc_norm_final_pos", "y"): np.nan,
            ("kc_norm_final_pos", "z"): np.nan,
            ("panel_cm_final_pos", "x"): subgraph.nodes[root_id]["obj"].panel_cm[0],
            ("panel_cm_final_pos", "y"): subgraph.nodes[root_id]["obj"].panel_cm[1],
            ("panel_cm_final_pos", "z"): subgraph.nodes[root_id]["obj"].panel_cm[2],
        })
        next_panel_misalignments(root_id, None)

        return result_table


        
def demo_kc_maxwell_creation():
    big_kc = MaxwellKinematicCoupling("big_kc", np.array([0, 0, 0]), np.array([0, 0, 1]), 10, 5, -5)
    short_kc = MaxwellKinematicCoupling("short_kc", np.array([0, 0, 0]), np.array([0, 0, 1]), 10, 3, -3)
    small_kc = MaxwellKinematicCoupling("small_kc", np.array([0, 0, 0]), np.array([0, 0, 1]), 4, 2, -2)

    misalignment_trans = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    misalignment_rot = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    np.random.seed(42)
    misalignment_rand = 2 * np.random.rand(6) - 1
    
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(10, 9))
    view_angle = (15, 75)  # Set a consistent view angle for all subplots

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    ax.set_title("Big Maxwell Translational Misalignment", fontsize=9)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    ax.set_title("Big Maxwell Rotational Misalignment", fontsize=9)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    big_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    ax.set_title("Big Maxwell Random Misalignment", fontsize=9)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    ax.set_title("Short Maxwell Translational Misalignment", fontsize=9)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    ax.set_title("Short Maxwell Rotational Misalignment", fontsize=9)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    short_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    ax.set_title("Short Maxwell Random Misalignment", fontsize=9)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_trans, ax=ax)
    ax.set_title("Small Maxwell Translational Misalignment", fontsize=9)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_rot, ax=ax)
    ax.set_title("Small Maxwell Rotational Misalignment", fontsize=9)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    small_kc.plot_solve_kc_pose(misalignment_rand, ax=ax)
    ax.set_title("Small Maxwell Random Misalignment", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()


if __name__ == "__main__":
    demo_kc_maxwell_creation()
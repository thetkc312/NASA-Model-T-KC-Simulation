import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

from chatgpt_demo import estimate_pose, plot_pose_estimation
from itertools import product




class KinematicCoupling():
    """
    A class to represent a kinematic coupling. It solves for the misalignment poses of the panel based on all combinations of misalignment values.
    """
    def __init__(self, kc_id, kc_contact_points: np.ndarray[6, 3], kc_contact_norms: np.ndarray[6, 3], kc_origin: np.ndarray[3], kc_origin_norm: np.ndarray[3], kc_length: float = None, stowed_arm_protrusion: float = None):
        """
        Add information about a kinematic coupling for a panel.
        :param kc_contact_points: 6 contact point coordinates of the kinematic coupling in the kc coordinate frame*
        :param kc_contact_norms: 6 unit normals corresponding to planar contact surfaces in the kinematic coupling, in the kc coordinate frame*
        :param kc_origin: origin of the kinematic coupling in the array coordinate frame
        :param kc_origin_norm: normal of the kinematic coupling in the array coordinate frame
        :param kc_length: length of the kinematic coupling
        :param stowed_arm_protrusion: stowed arm angle from the vertical when stowed
        :param misalignment_array: array of misalignment values, each corresponding to translation of a kc contact plane along its normal
        :param misalignment_poses: array of estimated poses corresponding to the misalignment values

        *Note: the contact points and normals are in the kinematic coupling local coordinate frame, which is defined as follows:
        - The coordinate frame origin is at the center of the kinematic coupling
        - The x axis is the direction of the kc_origin_norm
        - The y axis is the direction of the length of the kinematic coupling
        - The z axis is orthogonal to the x and y axes
        """
        self.kc_id = kc_id
        self.kc_contact_points = kc_contact_points
        self.kc_contact_norms = kc_contact_norms
        self.kc_origin = kc_origin
        self.kc_origin_norm = kc_origin_norm
        self.kc_length = kc_length
        self.stowed_arm_protrusion = stowed_arm_protrusion

    def solve_pose(self, misalignment: np.ndarray[6]):
        """
        Solve for the misalignment pose of the panel based on a single set of misalignment values. Return and do not save the result.
        :param misalignment: row of 6 misalignment values to be evaluated, each corresponding to translation of a kc contact plane along its normal
        """
        result = np.array(estimate_pose(misalignment).x)
        return result

    


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

    def get_data_post_kc_misalignment(self, kc_id: str, misalignments: np.ndarray[6], external_rotation: R = None, external_translation: np.ndarray[3] = None, linked_kc_origins: np.ndarray[:, 3] = None) -> tuple:
        """
        After applying each evaluated misalignment for some kinematic coupling, get more information about the panel based on the misalignment pose.
        :param kc_id: id of the kinematic coupling
        :param linked_kc_origins: array of origins of the kinematic couplings linked to the panel (optional)
        :param misalignments: array of misalignment values, each corresponding to translation of a kc contact plane along its normal
        :param external_rotation: rotation of the panel in the array coordinate frame derived from its parent (optional)
        :param external_translation: translation of the panel in the array coordinate frame derived from its parent (optional)
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
        kc_to_linked_kcs = linked_kc_origins - kc_origin if linked_kc_origins is not None else None

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
        local_translation = external_rotation.apply(array_translation) if external_rotation else array_translation

        # Apply the rotations and translations
        resulting_kc_origin = kc_origin + local_translation + (external_translation if external_translation is not None else 0)
        resulting_cm = net_rotation.apply(kc_to_cm) + resulting_kc_origin
        resulting_kc_norm = net_rotation.apply(kc_to_norm) + resulting_kc_origin

        if linked_kc_origins is not None:
            resulting_linked_kc_shifts = net_rotation.apply(kc_to_linked_kcs) + resulting_kc_origin - linked_kc_origins
            return resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation, resulting_linked_kc_shifts
        else:
            return resulting_kc_origin, resulting_cm, resulting_kc_norm

class GroundPanel():
    """
    A class to represent a ground panel object. This is a special type of panel that is not linked to any other panels.
    """
    def __init__(self, ground_id, ground_cm):
        self.ground_id = ground_id
        self.ground_cm = ground_cm


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
        self.kc_id_to_edge = {}  # Map kc_id to (parent_panel_id, child_panel_id)

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
        if kc.kc_id in self.kc_id_to_edge:
            raise ValueError(f"Kinematic coupling with id {kc.kc_id} already exists.")

        # Add the kinematic coupling as an edge in the graph
        self.graph.add_edge(parent_panel_id, child_panel_id, obj=kc, type="kc", kc_id=kc.kc_id)

        # Map kc_id to the edge
        self.kc_id_to_edge[kc.kc_id] = (parent_panel_id, child_panel_id)

        # Add the kinematic coupling to the child panel
        child_panel = self.graph.nodes[child_panel_id]["obj"]
        child_panel.add_panel_kc(kc)

    def get_kc_path(self, kc_ids: list[str]) -> tuple[str, nx.DiGraph]:
        """
        Propagate the misalignment poses for multiple kinematic couplings based on all combinations of misalignment values.
        :param kc_ids: variable number of kinematic coupling ids to propagate
        """
        for kc_id in kc_ids:
            # Retrieve the edge using kc_id
            if kc_id not in self.kc_id_to_edge:
                raise ValueError(f"Kinematic coupling with id {kc_id} not found.")
            parent_panel_id, child_panel_id = self.kc_id_to_edge[kc_id]
            kc = self.graph[parent_panel_id][child_panel_id]["obj"]
            
        # Check if the edges form a continuous path
        nodes_in_path_ids = set()
        for kc_id in kc_ids:
            parent_panel_id, child_panel_id = self.kc_id_to_edge[kc_id]
            nodes_in_path_ids.add(parent_panel_id)
            nodes_in_path_ids.add(child_panel_id)

        subgraph = self.graph.subgraph(nodes_in_path_ids)
        # Ensure all edges form a continuous path
        if not nx.is_connected(subgraph.to_undirected()):
            raise ValueError("The provided kinematic couplings do not form a continuous path.")
        # Ensure that there are no cycles in the subgraph
        if not nx.is_directed_acyclic_graph(subgraph):
            raise ValueError("The provided kinematic couplings form a cycle. This is not allowed.")
        # Ensure that no two edges in the subgraph point to the same node
        for node_id in nodes_in_path_ids:
            if subgraph.in_degree(node_id) > 1:
                raise ValueError(f"Node with id {node_id} has more than one incoming edge. This is not allowed.")

        # Find the lowest root node
        root_candidates_ids = [node_id for node_id in nodes_in_path_ids if self.graph.in_degree(node_id) == 0]
        if len(root_candidates_ids) != 1:
            raise ValueError("There is no unique root node for the provided kinematic couplings. In other words, there is no clear" \
            "starting point to evaluate misalignment propagation from for the panels linked by the provided kinematic couplings.")
        lowest_root_node_id = root_candidates_ids[0]

        return lowest_root_node_id, subgraph
    
    def propagate_misalignment_poses(self, root_id, subgraph, misalignment_dict: dict[str: np.ndarray[6]]):
        """
        Propagate the misalignment poses for multiple kinematic couplings based on all combinations of misalignment values.
        :param root_id: id of the root node
        :param subgraph: subgraph containing the nodes and edges to propagate
        :param misalignments: array of misalignment values, each corresponding to translation of a kc contact plane along its normal
        """
        if misalignments.shape[0] != len(subgraph.edges):
            raise ValueError("The number of misalignment values must match the number of kinematic couplings in the subgraph.")

        # FIXME: This is an incomplete and incorrect solution to get the misalignment values for each kinematic coupling
        def next_panel_misalignments(parent_panel_id, ext_rot, ext_trans, linked_kc_shifts):
            for child_panel_id, linked_kc_shift in zip(subgraph.successors(parent_panel_id), linked_kc_shifts):
                # Get the panel object
                child_panel = self.graph.nodes[child_panel_id]["obj"]
                # Get the kinematic coupling id
                kc_id = subgraph[root_id][child_panel_id]["kc_id"]
                linked_kc_ids = [subgraph[child_panel_id][grandchild_panel]["kc_id"] for grandchild_panel in subgraph.successors(child_panel_id)]
                linked_kc_origins = np.array([
                    subgraph[child_panel_id][self.kc_id_to_edge[linked_kc_id][1]]["obj"].kc_origin
                    for linked_kc_id in linked_kc_ids
                ])
                linked_kc_origins += linked_kc_shifts
                resulting_kc_origin, resulting_cm, resulting_kc_norm, net_rotation, resulting_linked_kc_shifts = child_panel.get_data_post_kc_misalignment(kc_id, misalignment_dict[kc_id], linked_kc_origins=linked_kc_origins)
                # Update the net rotation and linked kinematic coupling shifts
                next_panel_misalignments(child_panel_id, net_rotation, resulting_linked_kc_shifts)


        



if __name__ == "__main__":
    # Example usage
    panel = Panel(panel_id=1)
    kc_contact_points = np.array([
        [1, -0.1, 0.0],
        [1, 0.1, 0.0],
        [-1.1, 1, 0.0],
        [-0.9, 1, 0.0],
        [-1.1, -1, 0.0],
        [-0.9, -1, 0.0]
    ])
    kc_contact_norms = np.array([
        [0, 1, 1],
        [0, -1, 1],
        [1, 0, 1],
        [-1, 0, 1],
        [1, 0, 1],
        [-1, 0, 1]
    ], dtype=float)
    kc_origin = np.array([0.0, 0.0, 0.0])
    kc_origin_norm = np.array([1.0, 0.0, 0.0])
    kc_length = 2.5

    panel.add_panel_kc(kc_contact_points=kc_contact_points,
                       kc_contact_norms=kc_contact_norms,
                       kc_origin=kc_origin,
                       kc_origin_norm=kc_origin_norm)

    misalignments = [
        [-1.0, 0.0, 1.0],
        [0.0, 1.0],
        0.0,
        0.0,
        [-0.5, 0.5],
        0.0
    ]

    panel.solve_misalignment_poses(misalignments)
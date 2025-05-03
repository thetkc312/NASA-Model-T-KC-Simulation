import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

from chatgpt_demo import estimate_pose, plot_pose_estimation
from itertools import product


class KinematicCoupling():
    """
    A class to represent a kinematic coupling.
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
        self.misalignment_array = None
        self.misalignment_poses = None
        

    def solve_misalignment_poses(self, misalignments: list[list[float] | float]):
        """
        Solve for the misalignment poses of the panel based on all combinations of misalignment values.
        :param misalignments: 6 sets of misalignment values to be evaluated, each corresponding to translation of the corresponding kc contact plane along its normal
        """
        # Ensure the misalignments are in the correct format
        if len(misalignments) != 6:
            raise ValueError("Misalignments must be a list of 6 sets of values.")

        # Ensure all items in misalignments are lists for uniform processing
        misalignments = [
            [m] if isinstance(m, float) else m for m in misalignments
        ]

        # Use itertools.product to generate the Cartesian product of all misalignment values
        combinations = list(product(*misalignments))

        # Convert the combinations into a numpy array
        misalignment_array = np.array(combinations)
        self.misalignment_array = misalignment_array

        # Apply the estimate_pose function to each row of misalignment_array
        results = np.array([estimate_pose(misalignment).x for misalignment in misalignment_array])

        # Store the results in attributes for further use
        self.misalignment_poses = results

    


class Panel():
    """
    A class to represent a panel object. 
    """
    def __init__(self, panel_id, panel_cm = None):
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

    def solve_kc_misalignment_poses(self, kc_id: str, misalignments: list[list[float] | float]):
        """
        Solve for the misalignment poses of the panel based on all combinations of misalignment values.
        :param kc_id: id of the kinematic coupling
        :param misalignments: 6 sets of misalignment values to be evaluated, each corresponding to translation of the corresponding kc contact plane along its normal
        """
        # Find the kinematic coupling object with the given id
        kc_object = self.kinematic_couplings.get(kc_id) if kc_id in self.kinematic_couplings else None
        if kc_object is None:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found.")

        # Solve for the misalignment poses
        kc_object.solve_misalignment_poses(misalignments)

    def get_data_post_kc_misalignment(self, kc_id: str, linked_kc_origins: np.ndarray[:, 3]) -> tuple[np.ndarray[:, 3], np.ndarray[:, 3], np.ndarray[:, 3], np.ndarray[:, 3, :]]:
        """
        After applying each evaluated misalignment for some kinematic coupling, get more information about the panel based on the misalignment pose.
        :param kc_id: id of the kinematic coupling
        :param linked_kc_origins: array of linked kinematic coupling origins in array coordinates, to be used for their external misalignment in subsequent steps
        :return: tuple of resulting center of mass positions, kc origin positions, kc norm positions and linked kc origins in array coordinates
        """
        # Find the kinematic coupling object with the given id
        if kc_id not in self.kinematic_couplings:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found in set of kinematic couplings.")
        kc_object = self.kinematic_couplings[kc_id]

        # Ensure the panel center of mass is defined
        if self.panel_cm is None:
            raise ValueError("Panel center of mass (panel_cm) is not defined.")

        # Convert kc_origin_norm to a unit vector
        kc_x_axis = kc_object.kc_origin_norm / np.linalg.norm(kc_object.kc_origin_norm)

        # Define the panel coordinate system axes in array coordinates
        kc_z_axis = np.array([0, 0, 1])  # Assuming array z-axis is aligned with panel z-axis
        kc_y_axis = np.cross(kc_z_axis, kc_x_axis)
        kc_y_axis /= np.linalg.norm(kc_y_axis)
        kc_z_axis = np.cross(kc_x_axis, kc_y_axis)

        # Construct the rotation matrix from panel coordinates to array coordinates
        panel_to_array_rotation = np.column_stack((kc_x_axis, kc_y_axis, kc_z_axis))

        # Extract misalignment poses (rotation and translation vectors in panel coordinates)
        misalignment_poses = kc_object.misalignment_poses  # Shape: [n, 6]
        misalignment_rotations = misalignment_poses[:, :3]  # [:, :3] rotation vectors in panel coordinates
        misalignment_translations = misalignment_poses[:, 3:]  # [:, 3:] translation vectors in panel coordinates

        # Transform all misalignment rotations and translations into array coordinates
        rotations_array_coords = misalignment_rotations @ panel_to_array_rotation.T
        translations_array_coords = misalignment_translations @ panel_to_array_rotation.T

        # Initialize the resulting center of mass and kc norm positions as numpy arrays
        resulting_cm_positions = np.zeros((misalignment_poses.shape[0], 3))
        resulting_kc_origin_positions = translations_array_coords + kc_object.kc_origin
        resulting_kc_norm_positions = np.zeros((misalignment_poses.shape[0], 3))
        resulting_linked_kc_positions = np.zeros((misalignment_poses.shape[0], 3, linked_kc_origins.shape[0]))

        # Apply misalignment rotations and translations
        for i, (rotation_vector_coords, translation_vector) in enumerate(zip(rotations_array_coords, translations_array_coords)):
            # Apply the rotation to the panel center of mass relative to kc_origin
            rotation_vector = R.from_rotvec(rotation_vector_coords).as_rotvec()
            rotated_cm = rotation_vector.apply(self.panel_cm - kc_object.kc_origin) + kc_object.kc_origin
            rotated_kc_norm = rotation_vector.apply(kc_object.kc_origin_norm - kc_object.kc_origin) + kc_object.kc_origin
            rotated_linked_kc = rotation_vector.apply(linked_kc_origins - kc_object.kc_origin) + kc_object.kc_origin

            # Apply the translation to the rotated center of mass
            translated_cm = rotated_cm + translation_vector
            translated_kc_norm = rotated_kc_norm + translation_vector
            translated_linked_kc = rotated_linked_kc + translation_vector

            # Store the resulting center of mass position in the array
            resulting_cm_positions[i] = translated_cm
            resulting_kc_norm_positions[i] = translated_kc_norm
            resulting_linked_kc_positions[i] = translated_linked_kc.T
            

        return resulting_cm_positions, resulting_kc_origin_positions, resulting_kc_norm_positions, resulting_linked_kc_positions
    

    def get_data_post_external_misalignment(self, external_rot: R, external_trans: np.ndarray[3], kc_id: str, kc_misalignment_cm_positions, kc_misalignment_origin_positions, kc_misalignment_norm_positions, linked_kc_origin_positions) -> tuple[np.ndarray[:, 3], np.ndarray[:, 3], np.ndarray[:, 3], np.ndarray[:, 3, :]]:
        """
        Apply external rotation and translation to the misalignment data.
        :param external_rot: Rotation object representing the external rotation
        :param external_trans: Translation vector representing the external translation
        :param kc_id: id of the kinematic coupling
        :param kc_misalignment_cm_positions: Center of mass positions after misalignment
        :param kc_misalignment_origin_positions: KC origin positions after misalignment
        :param kc_misalignment_norm_positions: KC norm positions after misalignment
        :param resulting_linked_kc_positions: Linked KC positions after misalignment
        :return: tuple of resulting center of mass positions, KC origin positions, and KC norm positions
        """
        if kc_id not in self.kinematic_couplings:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found in set of kinematic couplings.")
        
        # Find the kinematic coupling object with the given id
        kc_object = self.kinematic_couplings[kc_id]

        # Apply the external rotation relative to kc_origin
        resulting_cm_positions = external_rot.apply(kc_misalignment_cm_positions - kc_object.kc_origin) + kc_object.kc_origin
        resulting_kc_origin_positions = external_rot.apply(kc_misalignment_origin_positions - kc_object.kc_origin) + kc_object.kc_origin
        resulting_kc_norm_positions = external_rot.apply(kc_misalignment_norm_positions - kc_object.kc_origin) + kc_object.kc_origin
        resulting_linked_kc_positions = np.zeros((linked_kc_origin_positions.shape[0], linked_kc_origin_positions.shape[1], linked_kc_origin_positions.shape[2]), dtype=float)
        for i in range(resulting_linked_kc_positions.shape[2]):
            resulting_linked_kc_positions[:, :, i] = external_rot.apply(resulting_linked_kc_positions[:, :, i] - kc_object.kc_origin) + kc_object.kc_origin

        # Apply the external translation
        resulting_cm_positions += external_trans
        resulting_kc_origin_positions += external_trans
        resulting_kc_norm_positions += external_trans
        for i in range(resulting_linked_kc_positions.shape[2]):
            resulting_linked_kc_positions[:, :, i] += external_trans

        return resulting_cm_positions, resulting_kc_origin_positions, resulting_kc_norm_positions, resulting_linked_kc_positions
    

class GroundPanel():
    """
    A class to represent a ground panel object. This is a special type of panel that is not linked to any other panels.
    """
    def __init__(self, ground_id, ground_cm):
        self.ground_id = ground_id
        self.ground_cm = ground_cm


class LinkedPanelArray():
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

    def evaluate_kc_misalignments(self, misalignments: list[list[float] | float], *kc_ids: str):
        """
        Evaluate the misalignment poses for multiple kinematic couplings based on all combinations of misalignment values.
        :param misalignments: 6 sets of misalignment values to be evaluated, each corresponding to translation of the corresponding kc contact plane along its normal
        :param kc_ids: variable number of kinematic coupling ids to evaluate
        """
        for kc_id in kc_ids:
            # Retrieve the edge using kc_id
            if kc_id not in self.kc_id_to_edge:
                raise ValueError(f"Kinematic coupling with id {kc_id} not found.")
            parent_panel_id, child_panel_id = self.kc_id_to_edge[kc_id]
            kc = self.graph[parent_panel_id][child_panel_id]["obj"]

            # Solve for the misalignment poses
            kc.solve_misalignment_poses(misalignments)

    def evaluate_kc_pose(self, kc_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Based on previously solved misalignment poses for a single kinematic coupling, get more information about the panel.
        :param kc_id: id of the kinematic coupling
        :return: tuple of resulting center of mass positions, kc origin positions, and kc norm positions in array coordinates
        """
        # Retrieve the edge using kc_id
        if kc_id not in self.kc_id_to_edge:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found.")
        parent_panel_id, child_panel_id = self.kc_id_to_edge[kc_id]
        kc = self.graph[parent_panel_id][child_panel_id]["obj"]

        # Get the data post misalignment
        child_panel = self.graph.nodes[child_panel_id]["obj"]
        return child_panel.get_data_post_kc_misalignment(kc_id)

    def propagate_kc_poses(self, *kc_ids: str):
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

            if kc.misalignment_poses is None:
                raise ValueError(f"Kinematic coupling with id {kc_id} has not been solved for misalignment poses.")
        # Check if the edges form a continuous path
        nodes_in_path_ids = set()
        for kc_id in kc_ids:
            if kc_id not in self.kc_id_to_edge:
                raise ValueError(f"Kinematic coupling with id {kc_id} not found.")
            parent_panel_id, child_panel_id = self.kc_id_to_edge[kc_id]
            nodes_in_path_ids.add(parent_panel_id)
            nodes_in_path_ids.add(child_panel_id)

        # Ensure all edges form a continuous path
        subgraph = self.graph.subgraph(nodes_in_path_ids)
        if not nx.is_connected(subgraph.to_undirected()):
            raise ValueError("The provided kinematic couplings do not form a continuous path.")

        # Find the lowest root node
        root_candidates_ids = [node_id for node_id in nodes_in_path_ids if self.graph.in_degree(node_id) == 0]
        if len(root_candidates_ids) != 1:
            raise ValueError("There is no unique root node for the provided kinematic couplings. In other words, there is no clear" \
            "starting point to evaluate misalignment propagation from for the panels linked by the provided kinematic couplings.")
        lowest_root_node_id = root_candidates_ids[0]

        # FIXME: Loop through the nodes to which the lowest root node points, with edges in the subgraph
        for child_node_id in subgraph.successors(lowest_root_node_id):
            edge_data = self.graph[lowest_root_node_id][child_node_id]
            if edge_data["type"] == "kc":
                kc = edge_data["obj"]
            if kc.misalignment_poses is None:
                raise ValueError(f"Kinematic coupling with id {kc.kc_id} has not been solved for misalignment poses.")
            # Propagate poses to the child panel
            child_panel = self.graph.nodes[child_node_id]["obj"]
            parent_panel = self.graph.nodes[lowest_root_node_id]["obj"]
            linked_kc_origins = np.array([
                self.graph[child_node_id][successor]["obj"].kc_origin
                for successor in subgraph.successors(child_node_id)
                if self.graph[child_node_id][successor]["type"] == "kc"
                ])
            resulting_cm_positions, resulting_kc_origin_positions, resulting_kc_norm_positions, resulting_linked_kc_positions = child_panel.get_data_post_kc_misalignment(kc.kc_id, linked_kc_origins)


        



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
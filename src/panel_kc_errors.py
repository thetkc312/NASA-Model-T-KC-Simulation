import numpy as np
from scipy.spatial.transform import Rotation as R

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
        

    def solve_misalignment_poses(self, misalignments: list[list[float] | float]):
        """
        Solve for the misalignment poses of the panel based on all combinations of misalignment values.
        :param misalignments: 6 sets of misalignment values to be evaluated, each corresponding to translation of the corresponding kc contact plane along its normal
        """
        # Generate all possible combinations of misalignment values

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

    


class PanelObject():
    """
    A class to represent a panel object. 
    """
    def __init__(self, panel_id, panel_cm = None):
        self.panel_id = panel_id
        self.panel_cm = panel_cm
        self.kinematic_couplings = dict()
        self.external_rot_trans = dict()

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

    def add_external_rot_trans(self, kc_id: str, rot_trans_id:str, external_rot_trans: np.ndarray):
        """
        Add or override an external rotation and translation to the panel at the point of some kinematic coupling. This
        represents the resultant rotation and translation of the panel kc based on the panel to which it is mounted.
        :param kc_id: id of the kinematic coupling where the external rotations and translations are applied
        :param external_rot_trans: 6-element array containing the rotation vector (about the corresponding kc_origin) and translation vector
        """
        if kc_id not in self.kinematic_couplings:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found in set of kinematic couplings.")
        self.external_rot_trans[kc_id] = external_rot_trans

    def get_cm_post_kc_misalignment(self, kc_id: str):
        """
        Get all resulting centers of mass of the panel after applying each evaluated misalignment for some kinematic coupling.
        :param kc_id: id of the kinematic coupling
        :return: np.ndarray of resulting panel center of mass positions in array coordinates
        """
        # Find the kinematic coupling object with the given id
        kc_object = self.kinematic_couplings.get(kc_id)
        if kc_object is None:
            raise ValueError(f"Kinematic coupling with id {kc_id} not found in set of kinematic couplings.")

        # Ensure the panel center of mass is defined
        if self.panel_cm is None:
            raise ValueError("Panel center of mass (panel_cm) is not defined.")

        # Extract misalignment poses (rotation and translation vectors in panel coordinates)
        misalignment_poses = kc_object.misalignment_poses  # Shape: [n, 6]
        misalignment_rotations = misalignment_poses[:, :3]  # [:, :3] rotation vectors in panel coordinates
        misalignment_translations = misalignment_poses[:, 3:]  # [:, 3:] translation vectors in panel coordinates

        # Convert kc_origin_norm to a unit vector
        kc_x_axis = kc_object.kc_origin_norm / np.linalg.norm(kc_object.kc_origin_norm)

        # Define the panel coordinate system axes in array coordinates
        kc_z_axis = np.array([0, 0, 1])  # Assuming array z-axis is aligned with panel z-axis
        kc_y_axis = np.cross(kc_z_axis, kc_x_axis)
        kc_y_axis /= np.linalg.norm(kc_y_axis)
        kc_z_axis = np.cross(kc_x_axis, kc_y_axis)

        # Construct the rotation matrix from panel coordinates to array coordinates
        panel_to_array_rotation = np.column_stack((kc_x_axis, kc_y_axis, kc_z_axis))

        # Initialize the resulting center of mass positions as a numpy array
        resulting_cm_positions = np.zeros((misalignment_poses.shape[0], 3))

        # Step 1: Apply misalignment rotations and translations
        for i, (rotation_vec, translation_vec) in enumerate(zip(misalignment_rotations, misalignment_translations)):
            # Convert rotation vector from panel to array coordinates
            rotation_array_coords = panel_to_array_rotation @ rotation_vec

            # Convert translation vector from panel to array coordinates
            translation_array_coords = panel_to_array_rotation @ translation_vec

            # Apply the rotation to the panel center of mass relative to kc_origin
            rotated_cm = R.from_rotvec(rotation_array_coords).apply(self.panel_cm - kc_object.kc_origin) + kc_object.kc_origin

            # Apply the translation to the rotated center of mass
            translated_cm = rotated_cm + translation_array_coords

            # Store the resulting center of mass position in the array
            resulting_cm_positions[i] = translated_cm

        # Step 2: Apply external rotation and translation
        if kc_id in self.external_rot_trans:
            external_rot_trans = self.external_rot_trans[kc_id]
            external_rotation_vec = external_rot_trans[:3]  # [:3] rotation vector in array coordinates
            external_translation_vec = external_rot_trans[3:]  # [3:] translation vector in array coordinates

            # Apply the external rotation relative to kc_origin
            resulting_cm_positions = R.from_rotvec(external_rotation_vec).apply(resulting_cm_positions - kc_object.kc_origin) + kc_object.kc_origin

            # Apply the external translation
            resulting_cm_positions += external_translation_vec

        return resulting_cm_positions




class SolveMisalignmentPropagation():
    def __init__(self) -> None:
        pass

    def add_panel(self, panel):
        pass


if __name__ == "__main__":
    # Example usage
    panel = PanelObject(panel_id=1)
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
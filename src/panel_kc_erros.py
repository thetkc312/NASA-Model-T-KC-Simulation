import numpy as np

class PanelKC():
    def __init__(self, contact_points: np.ndarray[6, 3], contact_normals: np.ndarray[6, 3]):
        self.contact_points = contact_points
        self.contact_normals = contact_normals

    def get_misalignment_transformation(self, misalignments: np.ndarray[6,]) -> tuple[np.ndarray[3], np.ndarray[3, 3]]:
        """
        Get the transformation matrix and translation vector resulting from the specified
        misalignments of each contact point along its corresponding contact normal.
        
        :param misalignments: A 6-element array containing the misalignment values for each contact point.

        :return: A tuple containing the translation vector and the rotational transformation matrix.
        """
        # Consider a rigid body with 6 relevant points in 3D space, relative to the body's origin.
        # Each of those 6 points are restrained by only one degree of freedom. That degree of freedom
        # is along one translational axis, represented by constraint to a plane in 3D. Each point has
        # its own plane, and the normal of that plane is the direction of the restricted translational
        # degree of freedom. Consider now a scenario one of the planar constraints are very slightly 
        # moved in 3d space, along the normal of the plane. This is the misalignment of the constraint.
        # The misalignment is the distance between the original position of the constraint and the new
        # position.

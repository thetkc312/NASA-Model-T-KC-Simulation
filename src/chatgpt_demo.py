import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.set_printoptions(precision=4, suppress=True)

def estimate_pose(body_points, plane_normals, plane_misalignments):
    P = np.asarray(body_points)
    Nn = np.asarray(plane_normals)
    m = np.asarray(plane_misalignments)
    d = np.einsum('ij,ij->i', Nn, P) + m
    b = d - np.einsum('ij,ij->i', Nn, P)
    t0, *_ = np.linalg.lstsq(Nn, b, rcond=None)
    x0 = np.zeros(6)
    x0[3:] = t0

    def residual(x):
        rot_vec = x[:3]
        t = x[3:]
        rot = R.from_rotvec(rot_vec)
        Pw = rot.apply(P) + t
        return np.einsum('ij,ij->i', Nn, Pw) - d

    result = least_squares(residual, x0, method='lm')
    rot_opt = R.from_rotvec(result.x[:3])
    t_opt = result.x[3:]
    return rot_opt, t_opt

def generate_test_case(body_points, plane_normals, true_rot, true_t):
    transformed_points = true_rot.apply(body_points) + true_t
    d_true = np.einsum('ij,ij->i', plane_normals, transformed_points)
    d_original = np.einsum('ij,ij->i', plane_normals, body_points)
    misalignments = d_true - d_original
    return misalignments

def plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, true_rot=None, true_t=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Original and transformed points
    P_orig = np.array(body_points)
    P_est = est_rot.apply(P_orig) + est_t
    ax.scatter(*P_orig.T, color='blue', label='Original Body Points')
    ax.scatter(*P_est.T, color='red', label='Transformed (Estimated) Points')

    if true_rot is not None and true_t is not None:
        P_true = true_rot.apply(P_orig) + true_t
        ax.scatter(*P_true.T, color='green', label='Transformed (True) Points')

    # Visualize the planes as transparent squares centered on the projection of the estimated points onto the planes
    for i, (p, n, m) in enumerate(zip(P_orig, plane_normals, misalignments)):
        # Compute a point on the translated plane by projecting P_est onto the plane
        P_est_point = P_est[i]
        point_on_plane = P_est_point - np.dot(P_est_point - (p + m * n), n) * n
        # Generate two orthogonal vectors in the plane
        u = np.cross(n, np.array([1, 0, 0]))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, np.array([0, 1, 0]))
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        size = 0.1
        corners = [
            point_on_plane + size * (u + v),
            point_on_plane + size * (u - v),
            point_on_plane + size * (-u - v),
            point_on_plane + size * (-u + v)
        ]
        ax.add_collection3d(Poly3DCollection([corners], alpha=0.2, color='gray'))

    # Set uniform scale for all axes
    all_points = np.vstack([P_orig, P_est])
    if true_rot is not None and true_t is not None:
        all_points = np.vstack([all_points, P_true])
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Pose Estimation Visualization")

# No planes are misaligned
def test_case_0(body_points, plane_normals, ax):
    misalignments = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# All planes misaligned by 0.1 along their normals
def test_case_1(body_points, plane_normals, ax):
    misalignments = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# All planes misaligned by -0.1 along their normals
def test_case_2(body_points, plane_normals, ax):
    misalignments = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# The first ball-groove misaligned by 0.1 along its normal, all others unchanged (effectively 0)
def test_case_3(body_points, plane_normals, ax):
    misalignments = np.array([0.1, 0.1, 1e-6, 1e-6, 1e-6, 1e-6])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# The second ball-groove misaligned by 0.1 along its normal, all others unchanged (effectively 0)
def test_case_4(body_points, plane_normals, ax):
    misalignments = np.array([1e-6, 1e-6, 0.1, 0.1, 1e-6, 1e-6])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# The third ball-groove misaligned by 0.1 along its normal, all others unchanged (effectively 0)
def test_case_5(body_points, plane_normals, ax):
    misalignments = np.array([1e-6, 1e-6, 1e-6, 1e-6, 0.1, 0.1])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# All ball-grooves misaligned with alternating signs to induce a rotation
def test_case_6(body_points, plane_normals, ax):
    misalignments = np.array([0.1, -0.1, -0.1, 0.1, 0.1, -0.1])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# All planes misaligned with varying magnitudes to test robustness
def test_case_7(body_points, plane_normals, ax):
    misalignments = np.array([0.3, -0.2, 0.3, 0.5, 1e-6, -0.2])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# All planes misaligned with new varying magnitudes to test robustness
def test_case_8(body_points, plane_normals, ax):
    misalignments = np.array([-0.2, 1e-6, 0.1, -0.2, 0.3, 0.3])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, ax=ax)

# Generating misalignments based on a known true rotation and translation
# This test case is used to verify the accuracy of the pose estimation algorithm
def test_case_9(body_points, plane_normals, ax):
    true_rot = R.from_euler('xyz', [7, 5, 12], degrees=True)
    true_t = np.array([0.2, 0.4, 0.6])

    misalignments = generate_test_case(body_points, plane_normals, true_rot, true_t)
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)

    print("Estimated translation:", est_t)
    print("Estimated rotation matrix:")
    print(est_rot.as_matrix())

    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, true_rot, true_t, ax=ax)



if __name__ == "__main__":
    body_points = np.array([
        [1, -0.1, 0.0],
        [1, 0.1, 0.0],
        [-1.1, 1, 0.0],
        [-0.9, 1, 0.0],
        [-1.1, -1, 0.0],
        [-0.9, -1, 0.0]
    ])

    plane_normals = np.array([
        [0, 1, 1],
        [0, -1, 1],
        [1, 0, 1],
        [-1, 0, 1],
        [1, 0, 1],
        [-1, 0, 1]
    ], dtype=float)
    plane_normals /= np.linalg.norm(plane_normals, axis=1, keepdims=True)

    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(10, 9))
    view_angle = (45, 45)  # Set a consistent view angle for all subplots

    ax = axes[0, 0]
    ax.view_init(*view_angle)
    test_case_0(body_points, plane_normals, ax)
    ax.set_title("Test Case 0: No Misalignments", fontsize=9)

    ax = axes[0, 1]
    ax.view_init(*view_angle)
    test_case_1(body_points, plane_normals, ax)
    ax.set_title("Test Case 1: All Planes Misaligned by 0.1", fontsize=9)

    ax = axes[0, 2]
    ax.view_init(*view_angle)
    test_case_2(body_points, plane_normals, ax)
    ax.set_title("Test Case 2: All Planes Misaligned by -0.1", fontsize=9)

    ax = axes[1, 0]
    ax.view_init(*view_angle)
    test_case_3(body_points, plane_normals, ax)
    ax.set_title("Test Case 3: First Ball-Groove Misaligned by 0.1", fontsize=9)

    ax = axes[1, 1]
    ax.view_init(*view_angle)
    test_case_4(body_points, plane_normals, ax)
    ax.set_title("Test Case 4: Second Ball-Groove Misaligned by 0.1", fontsize=9)

    ax = axes[1, 2]
    ax.view_init(*view_angle)
    test_case_5(body_points, plane_normals, ax)
    ax.set_title("Test Case 5: Third Ball-Groove Misaligned by 0.1", fontsize=9)

    ax = axes[2, 0]
    ax.view_init(*view_angle)
    test_case_6(body_points, plane_normals, ax)
    ax.set_title("Test Case 6: Radial Misalignments", fontsize=9)

    ax = axes[2, 1]
    ax.view_init(*view_angle)
    test_case_7(body_points, plane_normals, ax)
    ax.set_title("Test Case 7: Varying Magnitudes 1", fontsize=9)

    ax = axes[2, 2]
    ax.view_init(*view_angle)
    test_case_8(body_points, plane_normals, ax)
    ax.set_title("Test Case 8: Varying Magnitudes 2", fontsize=9)

    for ax in axes.flat:
        ax.legend(fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.show()

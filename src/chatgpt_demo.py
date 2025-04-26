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

def plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, true_rot=None, true_t=None):
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

    # Visualize the planes as transparent squares centered on the constrained points
    for i, (p, n, m) in enumerate(zip(P_orig, plane_normals, misalignments)):
        # Compute a point on the translated plane
        point_on_plane = p + m * n
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
    plt.tight_layout()
    plt.show()

def test_case_1(body_points, plane_normals):
    true_rot = R.from_euler('xyz', [7, 5, 12], degrees=True)
    true_t = np.array([0.2, 0.4, 0.6])

    misalignments = generate_test_case(body_points, plane_normals, true_rot, true_t)
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)

    print("Estimated translation:", est_t)
    print("Estimated rotation matrix:")
    print(est_rot.as_matrix())

    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t, true_rot, true_t)

def test_case_2(body_points, plane_normals):
    misalignments = np.array([0.1, 0.1, -0.1, -0.1, 0.1, 0.2])
    est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)
    print("Estimated translation:", est_t)
    print("Estimated rotation matrix:")
    print(est_rot.as_matrix())

    plot_pose_estimation(body_points, plane_normals, misalignments, est_rot, est_t)


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
    #test_case_1(body_points, plane_normals)
    test_case_2(body_points, plane_normals)
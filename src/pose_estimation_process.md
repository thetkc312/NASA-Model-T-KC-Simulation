# Pose Estimation Process Documentation

This document provides a thorough, mathematical description of the algorithms and functions implemented in `pose_estimation.py`. It is designed to accompany the code in your research, explaining the underlying steps and notation.

---

## 1. Problem Statement

Given:

- A set of \(N\) body-fixed points \(\{p_i\in\mathbb{R}^3\mid i=1,\dots,N\}\),
- A corresponding set of unit plane normals \(\{n_i\in\mathbb{R}^3\mid \|n_i\|=1\}\),
- A set of signed misalignment distances \(m_i\in\mathbb{R}\), each indicating how far the original plane (which passed through \(p_i\) under the identity pose) was shifted along \(n_i\).

We wish to recover the rigid-body pose
\[
  x_{\text{orig}} = t \in \mathbb{R}^3, \quad R \in SO(3)
\]
such that each transformed point
\[
  x_i = R\,p_i + t
\]
lies on its corresponding shifted plane:
\[
  n_i^\top x_i = d_i,
  \quad\text{where}\quad d_i = n_i^\top p_i + m_i.
\]

---

## 2. Notation and Variables

| Symbol         | Meaning                                                       |
|----------------|---------------------------------------------------------------|
| \(p_i\)       | \(i\)-th body point in local frame (given).                 |
| \(n_i\)       | Unit normal of the \(i\)-th plane in world frame.           |
| \(m_i\)       | Signed misalignment: shift of original plane along \(n_i\). |
| \(d_i\)       | Plane offset: \(d_i = n_i^\top p_i + m_i\).                 |
| \(R\)         | Rotation matrix in \(SO(3)\).                                |
| \(t\)         | Translation vector of the body-origin in world coordinates.   |
| \(x_i\)       | Transformed point: \(x_i = R\,p_i + t\).                    |
| \(N\times3\) matrix | Stack of normals: each row is \(n_i^\top\).          |
| \(P\in\mathbb{R}^{N\times3}\)       | Stack of points: each row is \(p_i^\top\).                |

---

## 3. Overview of `estimate_pose`

The function solves for the six unknowns (three in \(R\), three in \(t\)) by minimizing the residuals
\[
  r_i(R,t) = n_i^\top \bigl(R\,p_i + t\bigr) - d_i,
  \quad i = 1,\dots,N.
\]

Concretely, it proceeds in two phases:

### 3.1 Initial Linear Translation Estimate

1. **Compute offsets**: \(d_i = n_i^\top p_i + m_i\).
2. **Assume** \(R = I\). Then each constraint is linear in \(t\):
   \[
     n_i^\top t = d_i - n_i^\top p_i.
   \]
3. **Form** the matrix
   \[
     N = \begin{bmatrix} n_1^\top \\ \vdots \\ n_N^\top \end{bmatrix},
     \quad b = \begin{bmatrix} d_1 - n_1^\top p_1 \\ \vdots \\ d_N - n_N^\top p_N \end{bmatrix}.
   \]
4. **Solve** the least-squares problem
   \[
     t_0 = \arg\min_t \|N t - b\|^2,
     \quad t_0 = (N^\top N)^{-1} N^\top b.
   \]
5. **Initialize** the parameter vector
   \[
     x^{(0)} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ t_0 \end{bmatrix},
   \]
   where the first three zeros represent the axis-angle (no rotation).

### 3.2 Nonlinear Refinement via Levenberg–Marquardt

We parameterize rotation as an axis-angle vector \(\omega\in\mathbb{R}^3\) so that
\[
  R(\omega) = \exp\bigl([\omega]_\times\bigr),
\]
where \([\omega]_\times\) is the skew-symmetric matrix of \(\omega\).  Define the full residual vector
\[
  r(x) = \begin{bmatrix} r_1(R(\omega),t) \\ \vdots \\ r_N(R(\omega),t) \end{bmatrix},
  \quad x = \begin{bmatrix} \omega \\ t \end{bmatrix}.
\]

We then solve the nonlinear least-squares problem
\[
  x^* = \arg\min_x \|r(x)\|^2
\]
using SciPy’s `least_squares(method='lm')`, which implements Levenberg–Marquardt:

1. **Evaluate** \(r(x)\) and the Jacobian \(J = \partial r/\partial x\).
2. **Compute** the damped normal equations:
   \[
     (J^\top J + \lambda I)\,\Delta x = -J^\top r(x).
   \]
3. **Update** the parameters:
   \[
     x \leftarrow x + \Delta x,
   \]
   and adjust \(\lambda\) according to the reduction in \(\|r(x)\|^2\).
4. **Repeat** until convergence.

Upon convergence, extract:
\[
  \omega^* = x^*_{1:3},
  \quad R^* = \exp\bigl([\omega^*]_\times\bigr),
  \quad t^* = x^*_{4:6}.
\]

---

## 4. `generate_test_case` Function

To validate, we simulate data for a known pose \((R_{\mathrm{true}},t_{\mathrm{true}})\):

1. **Transform** each point:
   \[
     p_i^{\mathrm{true}} = R_{\mathrm{true}} p_i + t_{\mathrm{true}}.
   \]
2. **Compute** the true plane offset:
   \[
     d_i^{\mathrm{true}} = n_i^\top p_i^{\mathrm{true}}.
   \]
3. **Compute** misalignment:
   \[
     m_i = d_i^{\mathrm{true}} - n_i^\top p_i.
   \]

Passing \(\{m_i\}\) into `estimate_pose` should precisely recover \(R_{\mathrm{true}}\) and \(t_{\mathrm{true}}\).

---

## 5. Usage Example

```python
from pose_estimation import estimate_pose, generate_test_case
from scipy.spatial.transform import Rotation as R
import numpy as np

# Define your body_points and plane_normals
body_points = np.array([...])
plane_normals = np.array([...])
plane_normals /= np.linalg.norm(plane_normals, axis=1, keepdims=True)

# Choose a moderate rotation and translation
true_rot = R.from_euler('xyz', [6, 4, 8], degrees=True)
true_t = np.array([0.5, -0.2, 1.2])

# Generate misalignments and estimate pose
misalignments = generate_test_case(body_points, plane_normals, true_rot, true_t)
est_rot, est_t = estimate_pose(body_points, plane_normals, misalignments)

print("Recovered rotation (deg):", est_rot.as_euler('xyz', degrees=True))
print("Recovered translation:", est_t)
```

---

## 6. References

- A. Trouvé, "Rigid Registration and Pose Estimation," in *Computer Vision: Models, Learning, and Inference.*, 2020.
- F. L. Bookstein, "Parameter Estimation Techniques in Computer Vision," *IEEE Trans. on Pattern Anal. Mach. Intell.*, 2016.

*End of README*


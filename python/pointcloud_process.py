import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore
import numpy as np
from sklearn.cluster import DBSCAN
import os
import random
import json
from datetime import datetime

# resolve directories from the active (or last) session
from helpers.session_manager import load_session

_session_name = input("Session name (blank = reuse last): ").strip() or None
_paths = load_session(".", _session_name)

# ---- Import directory ----
import_directory = str(_paths.merged_point_clouds)
filename = "merged01.ply"

# ---- Export directory ----
export_directory = str(_paths.exported_data)

# ---------- Constants ----------
EPS_NUM = 1e-8

# ---------- Segmentation Functions ----------

def preprocess_point_cloud(pcd, voxel_size=0.005, nb_neighbors=20, std_ratio=2.0):
    """
    Downsamples and denoises a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input cloud.
    voxel_size : float
        Voxel size (m) for grid downsampling.
    nb_neighbors : int
        Number of neighbors for statistical outlier removal.
    std_ratio : float
        Standard deviation ratio threshold for outlier removal.

    Returns
    -------
    o3d.geometry.PointCloud
        Processed (downsampled + denoised) cloud.
    """
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def detect_multiple_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000,
                           min_points=1000, max_planes=5):
    """
    Iteratively segments up to `max_planes` planar regions from a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input cloud to segment.
    distance_threshold : float
        RANSAC inlier distance threshold.
    ransac_n : int
        Number of points to sample for each RANSAC iteration.
    num_iterations : int
        Maximum RANSAC iterations.
    min_points : int
        Minimum inliers to accept a plane.
    max_planes : int
        Maximum number of planes to extract.

    Returns
    -------
    planes_for_cluster : list[o3d.geometry.PointCloud]
        Unpainted planes (preserve original colors) for later clustering.
    planes_for_display : list[o3d.geometry.PointCloud]
        Painted copies for visualization only.
    remaining : o3d.geometry.PointCloud
        Leftover (non-planar) points, unpainted.
    """
    planes_for_cluster = []
    planes_for_display = []
    remaining = pcd
    for _ in range(max_planes):
        if len(remaining.points) < min_points:
            break
        plane_model, inliers = remaining.segment_plane(distance_threshold, ransac_n, num_iterations)
        if len(inliers) < min_points:
            break

        inlier_cloud = remaining.select_by_index(inliers)  # keep true colors
        planes_for_cluster.append(inlier_cloud)

        painted = o3d.geometry.PointCloud(inlier_cloud)    # painted copy for display
        painted.paint_uniform_color([random.random(), random.random(), random.random()])
        planes_for_display.append(painted)

        remaining = remaining.select_by_index(inliers, invert=True)

    return planes_for_cluster, planes_for_display, remaining

def cluster_colored_plane(pcd_plane, eps=0.02, min_samples=350, color_weight=1.7, position_weight=0.4):
    """
    Clusters a single plane using DBSCAN on [position*weight | color*weight] features.

    Parameters
    ----------
    pcd_plane : o3d.geometry.PointCloud
        Points belonging to one extracted plane.
    eps : float
        DBSCAN neighborhood radius (in feature space).
    min_samples : int
        Minimum samples to form a cluster.
    color_weight : float
        Weight applied to RGB features.
    position_weight : float
        Weight applied to XYZ positions.

    Returns
    -------
    clusters_disp : list[o3d.geometry.PointCloud]
        Painted clusters for on-screen display.
    clusters_orig : list[o3d.geometry.PointCloud]
        Unpainted clusters (true colors) aligned with clusters_disp.
    """
    points = np.asarray(pcd_plane.points) * position_weight
    colors = np.asarray(pcd_plane.colors) * color_weight
    features = np.hstack((points, colors)).astype(np.float32)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_

    clusters_disp, clusters_orig = [], []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        indices = np.where(labels == cluster_id)[0]
        cluster_orig = pcd_plane.select_by_index(indices)
        clusters_orig.append(cluster_orig)

        cluster_disp = o3d.geometry.PointCloud(cluster_orig)
        cluster_disp.paint_uniform_color([random.random(), random.random(), random.random()])
        clusters_disp.append(cluster_disp)
    return clusters_disp, clusters_orig

# ---------- 2D Hull Helpers ----------

def _fit_plane_basis(points3d: np.ndarray):
    """
    Fits a local plane to 3D points via SVD and returns an orthonormal basis.

    Returns
    -------
    c : (3,) np.ndarray
        Centroid of points.
    u, v : (3,) np.ndarray
        In-plane orthonormal basis vectors.
    n : (3,) np.ndarray
        Plane normal (third singular vector).
    """
    c = points3d.mean(axis=0)
    P = points3d - c
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    u = Vt[0, :]
    v = Vt[1, :]
    n = Vt[2, :]
    return c, u, v, n

def _project_to_2d(points3d: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray):
    """
    Projects 3D points onto the local (u,v) plane coordinate system.

    Returns
    -------
    pts2 : (N,2) np.ndarray
        2D coordinates of points in the plane frame.
    """
    P = points3d - c
    x = P @ u
    y = P @ v
    return np.stack([x, y], axis=1)

def _monotone_chain_hull_indices(pts2: np.ndarray):
    """
    Computes indices of the 2D convex hull using the monotone chain algorithm.

    Parameters
    ----------
    pts2 : (N,2) np.ndarray
        2D points.

    Returns
    -------
    list[int]
        Indices (into pts2) of the hull polygon in order.
    """
    if len(pts2) < 3:
        return list(range(len(pts2)))
    order = np.lexsort((pts2[:, 1], pts2[:, 0]))
    pts = pts2[order]; idxs = order

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, lower_idx = [], []
    for p, i in zip(pts, idxs):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop(); lower_idx.pop()
        lower.append(p); lower_idx.append(i)

    upper, upper_idx = [], []
    for p, i in zip(reversed(pts), reversed(idxs)):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop(); upper_idx.pop()
        upper.append(p); upper_idx.append(i)

    hull_idx = lower_idx[:-1] + upper_idx[:-1]
    seen, out = set(), []
    for i in hull_idx:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def create_2d_hull_lineset_from_cluster(cluster: o3d.geometry.PointCloud):
    """
    Builds a LineSet of the 2D convex hull (projected in-plane) of a cluster.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    o3d.geometry.LineSet | None
        LineSet for visualization or None if hull can't be formed.
    """
    pts = np.asarray(cluster.points)
    if pts.shape[0] < 3:
        return None
    c, u, v, _ = _fit_plane_basis(pts)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_local_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_local_idx) < 3:
        return None
    poly2 = pts2[hull_local_idx]
    poly3 = c + np.outer(poly2[:, 0], u) + np.outer(poly2[:, 1], v)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(poly3)
    lines = [[i, (i + 1) % len(poly3)] for i in range(len(poly3))]
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color([0, 0, 0])
    return ls

# ---- Extra helpers for hull-masked colors & centroids ----

def _polygon_centroid_2d(poly2: np.ndarray):
    """
    Computes centroid of a simple 2D polygon using the shoelace formula.

    Parameters
    ----------
    poly2 : (M,2) np.ndarray
        Polygon vertices in order.

    Returns
    -------
    (2,) np.ndarray
        Centroid in 2D.
    """
    x = poly2[:, 0]; y = poly2[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-12:
        return np.array([x.mean(), y.mean()], dtype=np.float32)
    Cx = (1.0 / (6.0 * A)) * np.sum((x + x1) * cross)
    Cy = (1.0 / (6.0 * A)) * np.sum((y + y1) * cross)
    return np.array([Cx, Cy], dtype=np.float32)

def _points_in_polygon(pts2: np.ndarray, poly2: np.ndarray):
    """
    Even–odd rule point-in-polygon test for a batch of 2D points.

    Parameters
    ----------
    pts2 : (N,2) np.ndarray
        Query points.
    poly2 : (M,2) np.ndarray
        Polygon vertices.

    Returns
    -------
    (N,) np.ndarray of bool
        Mask of points inside the polygon.
    """
    n = len(poly2)
    inside = np.zeros(len(pts2), dtype=bool)
    x = pts2[:, 0]; y = pts2[:, 1]
    px = poly2[:, 0]; py = poly2[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside

def get_hull_mask_and_centroid3d(cluster: o3d.geometry.PointCloud):
    """
    Computes a 2D-hull-based mask (in-plane) and its 3D centroid for a cluster.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    mask : (N,) np.ndarray of bool
        Points inside the 2D hull.
    centroid3d : (3,) np.ndarray
        3D centroid of the polygon, mapped back to 3D.
    """
    pts = np.asarray(cluster.points)
    if len(pts) < 3:
        return np.ones(len(pts), dtype=bool), pts.mean(axis=0)
    c, u, v, _ = _fit_plane_basis(pts)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_idx) < 3:
        return np.ones(len(pts), dtype=bool), pts.mean(axis=0)
    poly2 = pts2[hull_idx]
    mask = _points_in_polygon(pts2, poly2)
    c2d = _polygon_centroid_2d(poly2)
    centroid3d = c + c2d[0] * u + c2d[1] * v
    return mask, centroid3d

# ---- Helper used by export (poly + plane frame + centroid)
def compute_hull_poly2_poly3(cluster: o3d.geometry.PointCloud):
    """
    Computes the 2D hull in plane coordinates and maps it back to 3D.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    poly2 : (M,2) np.ndarray or None
        2D hull polygon in (u,v) coordinates.
    poly3 : (M,3) np.ndarray or None
        3D polygon points.
    c,u,v,n : (3,) np.ndarray
        Plane frame origin and basis.
    centroid3d : (3,) np.ndarray
        Centroid of the polygon in 3D.
    """
    pts = np.asarray(cluster.points)
    c, u, v, n = _fit_plane_basis(pts)
    if len(pts) < 3:
        return None, None, c, u, v, n, pts.mean(axis=0)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_idx) < 3:
        return None, None, c, u, v, n, pts.mean(axis=0)
    poly2 = pts2[hull_idx]
    poly3 = c + np.outer(poly2[:, 0], u) + np.outer(poly2[:, 1], v)
    c2d = _polygon_centroid_2d(poly2)
    centroid3d = c + c2d[0] * u + c2d[1] * v
    return poly2, poly3, c, u, v, n, centroid3d

# ---------- Color grouping helpers (post-hull) ----------

def _srgb_to_linear(c):
    """
    Converts sRGB components [0,1] to linear RGB.
    """
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def _rgb_to_lab(rgb):
    """
    Converts Nx3 sRGB array (0–1) to CIELAB (D65), returns Nx3 (L,a,b).
    """
    r, g, b = _srgb_to_linear(rgb[:, 0]), _srgb_to_linear(rgb[:, 1]), _srgb_to_linear(rgb[:, 2])
    RGB = np.stack([r, g, b], axis=1)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = RGB @ M.T
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = XYZ[:, 0] / Xn, XYZ[:, 1] / Yn, XYZ[:, 2] / Zn
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2) + 4/29)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=1)

def _cluster_color_feature(cluster: o3d.geometry.PointCloud, mode: str, use_hull_mask: bool):
    """
    Computes a representative color feature per cluster.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud
    mode : str
        "Lab a*b* only", "Raw", or "Chromaticity (L2)".
    use_hull_mask : bool
        If True, use only points inside the 2D hull.

    Returns
    -------
    (2,) or (3,) np.ndarray
        Feature vector depending on mode.
    """
    cols = np.asarray(cluster.colors).astype(np.float32)
    if cols.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if use_hull_mask:
        mask, _ = get_hull_mask_and_centroid3d(cluster)
        if mask.sum() >= 3:
            cols = cols[mask]
    mode_l = (mode or "Lab a*b* only").strip().lower()
    if mode_l.startswith("lab"):
        Lab = _rgb_to_lab(cols)
        return np.median(Lab[:, 1:3], axis=0).astype(np.float32)
    if mode_l.startswith("chrom"):
        norms = np.linalg.norm(cols, axis=1, keepdims=True)
        chrom = cols / (norms + EPS_NUM)
        return np.median(chrom, axis=0).astype(np.float32)
    return np.median(cols, axis=0).astype(np.float32)

def group_clusters_by_color(clusters_orig, mode="Lab a*b* only", eps_z=1.0, min_size=1,
                            use_hull_mask=False, centroids_out=None):
    """
    Groups clusters by color feature using DBSCAN over z-scored features.

    Parameters
    ----------
    clusters_orig : list[o3d.geometry.PointCloud]
        Unpainted clusters (true colors).
    mode : str
        Color feature mode (see _cluster_color_feature).
    eps_z : float
        DBSCAN eps in z-score space.
    min_size : int
        Minimum cluster size for DBSCAN.
    use_hull_mask : bool
        If True, compute features using points inside the 2D hull.
    centroids_out : list
        If provided, filled with per-cluster 3D centroids from hull.

    Returns
    -------
    labels : (K,) np.ndarray of int
    group_to_color : dict[int, np.ndarray]
        Mean RGB per DBSCAN group (0–1).
    report : list[str]
        Human-readable summary lines.
    """
    if not clusters_orig:
        return np.array([]), {}, ["Color Grouping Report", "No clusters."]
    feats, reps_rgb, centroids = [], [], []
    for cl in clusters_orig:
        cols_all = np.asarray(cl.colors).astype(np.float32)
        if use_hull_mask:
            mask, centroid3d = get_hull_mask_and_centroid3d(cl)
            centroids.append(centroid3d)
            cols_rep = cols_all[mask] if mask.sum() >= 3 else cols_all
        else:
            _, centroid3d = get_hull_mask_and_centroid3d(cl)
            centroids.append(centroid3d)
            cols_rep = cols_all
        reps_rgb.append(np.median(cols_rep, axis=0) if cols_rep.size else np.array([0.5, 0.5, 0.5], dtype=np.float32))
        feats.append(_cluster_color_feature(cl, mode, use_hull_mask))
    F = np.stack(feats, axis=0).astype(np.float32)
    mu = F.mean(axis=0, keepdims=True); sigma = F.std(axis=0, keepdims=True) + EPS_NUM
    Fz = (F - mu) / sigma
    db = DBSCAN(eps=float(eps_z), min_samples=int(min_size)).fit(Fz)
    labels = db.labels_
    group_to_color = {}
    for g in sorted(set(labels)):
        if g == -1:
            continue
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            continue
        mean_rgb = np.mean(np.stack([reps_rgb[i] for i in idx], axis=0), axis=0)
        group_to_color[g] = np.clip(mean_rgb, 0.0, 1.0)
    report = []
    report.append("Color Grouping Report")
    report.append(f"Mode: {mode} | eps(z): {eps_z:.2f} | min_size: {min_size} | hull_mask: {use_hull_mask}")
    report.append(f"Total clusters: {len(clusters_orig)}")
    for g in sorted(set(labels)):
        if g == -1:
            continue
        idx = np.where(labels == g)[0].tolist()
        rgb = group_to_color[g]
        report.append(f"- Group {g}: count={len(idx)}  meanRGB=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})  members={idx}")
    noise_idx = np.where(labels == -1)[0].tolist()
    if noise_idx:
        report.append(f"- Noise (-1): count={len(noise_idx)}  members={noise_idx}")
    report.append("Per-cluster 2D-hull centroids (3D coords):")
    for i, c3 in enumerate(centroids):
        report.append(f"  cluster[{i}]: ({c3[0]:.4f}, {c3[1]:.4f}, {c3[2]:.4f})")
    if centroids_out is not None:
        centroids_out.clear(); centroids_out.extend(centroids)
    return labels, group_to_color, report

# ---------- GUI Application ----------

class SegmentationApp:
    def __init__(self, pcd):
        """
        GUI application for multi-plane segmentation, clustering, hulls, grouping, and export.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud (meters).
        """
        self.original_pcd = pcd
        self.filtered_pcd = None
        self.processed_pcd = None

        # Planes
        self.planes_cluster = []   # unpainted (true colors)
        self.planes_display = []   # painted for visualization
        self.remaining_cloud = None

        # Clusters
        self.clusters = []              # DISPLAY clusters (painted)
        self.clusters_by_plane = {}     # idx -> list of display clusters
        self.cluster_originals = []     # UNPAINTED copies aligned with self.clusters
        self.cluster_disp_colors = []   # original display colors for "reset"

        # Hulls
        self.hull_lines = []            # 3D hulls (all)
        self.hull_selected = None
        self.hull_lines_2d = []         # 2D hulls (all)
        self.hull_selected_2d = None
        self.cluster_hull_centroids = []    # centroid (3D) for each cluster's 2D hull

        # Report (only saved to file)
        self.last_report_lines = ["Color Grouping Report", "Use 'Save Report (.txt)' after grouping."]

        # Centroid visualization state
        self.centroid_spheres = []
        self.centroid_labels_enabled = False
        self.centroid_labels_data = []  # list of (pos3, text)

        # ---- Color grouping state (for JSON export) ----
        self.color_group_labels = None           # np.ndarray of ints (per-cluster group id; -1 = noise)
        self.color_group_mean_rgb = {}           # dict[int -> [r,g,b]]
        self.last_grouping_meta = None           # dict with settings + groups summary

        # Selections
        z_vals = np.asarray(pcd.points)[:, 2]
        self.min_z = float(z_vals.min()); self.max_z = float(z_vals.max())
        self.selected_plane_idx = 0; self.selected_cluster_idx = 0

        # Window & scene
        self.window = gui.Application.instance.create_window("Multi-Plane Segmentation", 1200, 820)

        # Left controls panel
        self.panel = gui.Vert(25, gui.Margins(10, 10, 10, 10))

        # Center scene
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        # Materials
        self.mat_points = rendering.MaterialRecord(); self.mat_points.shader = "defaultUnlit"; self.mat_points.point_size = 2.5
        self.mat_lines = rendering.MaterialRecord(); self.mat_lines.shader = "unlitLine"; self.mat_lines.line_width = 2.0
        self.mat_mesh = rendering.MaterialRecord(); self.mat_mesh.shader = "defaultLit"

        # Controls
        self.slider_min = gui.Slider(gui.Slider.DOUBLE)
        self.slider_max = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self.std_slider = gui.Slider(gui.Slider.DOUBLE)
        self.color_slider = gui.Slider(gui.Slider.DOUBLE)
        self.pos_slider = gui.Slider(gui.Slider.DOUBLE)
        self.eps_slider = gui.Slider(gui.Slider.DOUBLE)
        self.min_samples_slider = gui.Slider(gui.Slider.INT)
        self.dist_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.ransac_n_slider = gui.Slider(gui.Slider.INT)
        self.ransac_iter_slider = gui.Slider(gui.Slider.INT)

        self.plane_index_slider = gui.Slider(gui.Slider.INT)
        self.cluster_index_slider = gui.Slider(gui.Slider.INT)

        # Grouping controls
        self.group_mode_items = ["Lab a*b* only", "Raw", "Chromaticity (L2)"]
        self.group_mode_combo = gui.Combobox()
        for item in self.group_mode_items:
            self.group_mode_combo.add_item(item)
        self.group_mode_combo.selected_index = 0
        self.group_eps_slider = gui.Slider(gui.Slider.DOUBLE); self.group_eps_slider.set_limits(0.1, 3.0); self.group_eps_slider.double_value = 1.0
        self.group_min_slider = gui.Slider(gui.Slider.INT); self.group_min_slider.set_limits(1, 50); self.group_min_slider.int_value = 1
        self.group_use_hull_checkbox = gui.Checkbox("Use 2D hull colors only"); self.group_use_hull_checkbox.checked = True

        # Buttons
        self.btn_group_by_color = gui.Button("Group Clusters by Color"); self.btn_group_by_color.set_on_clicked(self.group_clusters_color_action)
        self.btn_reset_colors = gui.Button("Reset Display Colors"); self.btn_reset_colors.set_on_clicked(self.reset_display_colors)
        self.btn_save_report = gui.Button("Save Report (.txt)"); self.btn_save_report.set_on_clicked(self.save_report_to_txt)

        # Centroid buttons
        self.btn_show_centroids = gui.Button("Show Centroids (2D hull)"); self.btn_show_centroids.set_on_clicked(self.show_centroids)
        self.btn_hide_centroids = gui.Button("Hide Centroids"); self.btn_hide_centroids.set_on_clicked(self.hide_centroids)

        # ---- Export buttons (use EXPORT_DIR) ----
        self.export_dir = export_directory
        os.makedirs(self.export_dir, exist_ok=True)
        self.btn_export_selected_json = gui.Button("Export Selected Cluster (.json)")
        self.btn_export_selected_json.set_on_clicked(self.export_selected_cluster_json)
        self.btn_export_all_json = gui.Button("Export ALL Clusters (.json)")
        self.btn_export_all_json.set_on_clicked(self.export_all_clusters_json)

        # Defaults / limits
        self.slider_min.set_limits(self.min_z, self.max_z); self.slider_max.set_limits(self.min_z, self.max_z)
        self.slider_min.double_value = self.min_z; self.slider_max.double_value = self.max_z
        self.slider_min.set_on_value_changed(lambda _: self.apply_z_filter())
        self.slider_max.set_on_value_changed(lambda _: self.apply_z_filter())

        self.voxel_slider.set_limits(0.001, 0.05); self.voxel_slider.double_value = 0.005
        self.std_slider.set_limits(0.5, 5.0); self.std_slider.double_value = 2.0
        self.color_slider.set_limits(0.1, 10.0); self.color_slider.double_value = 1.7
        self.pos_slider.set_limits(0.1, 10.0); self.pos_slider.double_value = 0.4
        self.eps_slider.set_limits(0.001, 0.1); self.eps_slider.double_value = 0.02
        self.min_samples_slider.set_limits(10, 2000); self.min_samples_slider.int_value = 350
        self.dist_thresh_slider.set_limits(0.001, 0.1); self.dist_thresh_slider.double_value = 0.02
        self.ransac_n_slider.set_limits(3, 10); self.ransac_n_slider.int_value = 3
        self.ransac_iter_slider.set_limits(100, 5000); self.ransac_iter_slider.int_value = 1000

        self.plane_index_slider.set_limits(0, 0); self.plane_index_slider.int_value = 0
        self.cluster_index_slider.set_limits(0, 0); self.cluster_index_slider.int_value = 0

        # Workflow buttons
        self.btn_filter = gui.Button("Apply Z Filter"); self.btn_filter.set_on_clicked(self.apply_z_filter)
        self.btn_preprocess = gui.Button("Preprocess"); self.btn_preprocess.set_on_clicked(self.preprocess)
        self.btn_segment = gui.Button("Detect Planes"); self.btn_segment.set_on_clicked(self.detect_planes)
        self.btn_show_plane = gui.Button("Show Selected Plane"); self.btn_show_plane.set_on_clicked(self.show_selected_plane)
        self.btn_cluster_all = gui.Button("Cluster All Planes"); self.btn_cluster_all.set_on_clicked(self.cluster_planes_all)
        self.btn_cluster_selected = gui.Button("Cluster Selected Plane"); self.btn_cluster_selected.set_on_clicked(self.cluster_selected_plane)

        self.btn_hull_selected = gui.Button("Hull 3D: Selected Cluster"); self.btn_hull_selected.set_on_clicked(self.compute_hull_selected)
        self.btn_hull_all = gui.Button("Hull 3D: All Clusters"); self.btn_hull_all.set_on_clicked(self.compute_convex_hulls_all)
        self.btn_hull_selected_2d = gui.Button("Hull 2D: Selected Cluster"); self.btn_hull_selected_2d.set_on_clicked(self.compute_hull_selected_2d)
        self.btn_hull_all_2d = gui.Button("Hull 2D: All Clusters"); self.btn_hull_all_2d.set_on_clicked(self.compute_convex_hulls_all_2d)

        # ------- Layout of LEFT control panel (CollapsableVert sections) -------

        sect_gap = 8.0
        sect_marg = gui.Margins(0, 4, 0, 8)

        # 1) Z-Filtering
        sec1 = gui.CollapsableVert("1. Z-Filtering", sect_gap, sect_marg)
        sec1.add_child(gui.Label("Adjust min/max Z (m) and click 'Apply Z Filter'"))
        sec1.add_child(gui.Label("Z Min (m)"))
        sec1.add_child(self.slider_min)
        sec1.add_child(gui.Label("Z Max (m)"))
        sec1.add_child(self.slider_max)
        sec1.add_child(gui.Label(self.btn_filter.text))
        sec1.add_child(self.btn_filter)
        self.panel.add_child(sec1)

        # 2) Preprocessing
        sec2 = gui.CollapsableVert("2. Preprocessing", sect_gap, sect_marg)
        sec2.add_child(gui.Label("Voxel downsample + Remove outliers"))
        sec2.add_child(gui.Label("Voxel Size (m)"))
        sec2.add_child(self.voxel_slider)
        sec2.add_child(gui.Label("Outlier Std Ratio"))
        sec2.add_child(self.std_slider)
        sec2.add_child(gui.Label(self.btn_preprocess.text))
        sec2.add_child(self.btn_preprocess)
        self.panel.add_child(sec2)

        # 3) Plane Segmentation
        sec3 = gui.CollapsableVert("3. Plane Segmentation", sect_gap, sect_marg)
        sec3.add_child(gui.Label("RANSAC plane extraction"))
        sec3.add_child(gui.Label("Plane Distance Threshold (m)"))
        sec3.add_child(self.dist_thresh_slider)
        sec3.add_child(gui.Label("RANSAC N"))
        sec3.add_child(self.ransac_n_slider)
        sec3.add_child(gui.Label("RANSAC Iterations"))
        sec3.add_child(self.ransac_iter_slider)
        sec3.add_child(gui.Label(self.btn_segment.text))
        sec3.add_child(self.btn_segment)
        sec3.add_child(gui.Label("Selected Plane Index"))
        sec3.add_child(self.plane_index_slider)
        sec3.add_child(gui.Label(self.btn_show_plane.text))
        sec3.add_child(self.btn_show_plane)
        self.panel.add_child(sec3)

        # 4) Clustering
        sec4 = gui.CollapsableVert("4. Clustering", sect_gap, sect_marg)
        sec4.add_child(gui.Label("DBSCAN on [color*weight | position*weight]"))
        sec4.add_child(gui.Label("Color Weight"))
        sec4.add_child(self.color_slider)
        sec4.add_child(gui.Label("Position Weight"))
        sec4.add_child(self.pos_slider)
        sec4.add_child(gui.Label("DBSCAN eps"))
        sec4.add_child(self.eps_slider)
        sec4.add_child(gui.Label("Min Samples"))
        sec4.add_child(self.min_samples_slider)
        sec4.add_child(gui.Label(self.btn_cluster_all.text))
        sec4.add_child(self.btn_cluster_all)
        sec4.add_child(gui.Label(self.btn_cluster_selected.text))
        sec4.add_child(self.btn_cluster_selected)
        self.panel.add_child(sec4)

        # 5) Convex Hulls
        sec5 = gui.CollapsableVert("5. Convex Hulls", sect_gap, sect_marg)
        sec5.add_child(gui.Label("2D hulls in plane frame or 3D convex hulls"))
        sec5.add_child(gui.Label("Selected Cluster Index"))
        sec5.add_child(self.cluster_index_slider)
        sec5.add_child(gui.Label(self.btn_hull_selected.text))
        sec5.add_child(self.btn_hull_selected)
        sec5.add_child(gui.Label(self.btn_hull_all.text))
        sec5.add_child(self.btn_hull_all)
        sec5.add_child(gui.Label(self.btn_hull_selected_2d.text))
        sec5.add_child(self.btn_hull_selected_2d)
        sec5.add_child(gui.Label(self.btn_hull_all_2d.text))
        sec5.add_child(self.btn_hull_all_2d)
        self.panel.add_child(sec5)

        # 6) Color Grouping
        sec6 = gui.CollapsableVert("6. Color Grouping", sect_gap, sect_marg)
        sec6.add_child(gui.Label("Group clusters by color similarity"))
        sec6.add_child(gui.Label("Group Color Mode"))
        sec6.add_child(self.group_mode_combo)
        sec6.add_child(gui.Label("Group eps (z-score units)"))
        sec6.add_child(self.group_eps_slider)
        sec6.add_child(gui.Label("Group Min Size"))
        sec6.add_child(self.group_min_slider)
        sec6.add_child(gui.Label("Group Options"))
        sec6.add_child(self.group_use_hull_checkbox)
        sec6.add_child(gui.Label(self.btn_group_by_color.text))
        sec6.add_child(self.btn_group_by_color)
        sec6.add_child(gui.Label(self.btn_reset_colors.text))
        sec6.add_child(self.btn_reset_colors)
        self.panel.add_child(sec6)

        # 7) Centroids
        sec7 = gui.CollapsableVert("7. Centroids", sect_gap, sect_marg)
        sec7.add_child(gui.Label("Show/hide centroids of 2D hulls"))
        sec7.add_child(gui.Label(self.btn_show_centroids.text))
        sec7.add_child(self.btn_show_centroids)
        sec7.add_child(gui.Label(self.btn_hide_centroids.text))
        sec7.add_child(self.btn_hide_centroids)
        self.panel.add_child(sec7)

        # 8) Export
        sec8 = gui.CollapsableVert("8. Export", sect_gap, sect_marg)
        sec8.add_child(gui.Label(f"Export to JSON in folder: {self.export_dir}"))
        sec8.add_child(gui.Label(self.btn_export_selected_json.text))
        sec8.add_child(self.btn_export_selected_json)
        sec8.add_child(gui.Label(self.btn_export_all_json.text))
        sec8.add_child(self.btn_export_all_json)
        sec8.add_child(gui.Label(self.btn_save_report.text))
        sec8.add_child(self.btn_save_report)
        self.panel.add_child(sec8)        

        # Add widgets to window
        self.window.add_child(self.panel)
        self.window.add_child(self.scene)

        # Layout callback: left panel | scene
        self.window.set_on_layout(self.on_layout)

        # Scene background
        self.scene.scene.set_background([1, 1, 1, 1])

        # Initial view
        self.show_geometry(pcd)

    # ---------- Layout / Display helpers ----------

    def on_layout(self, context):
        """
        Places the left control panel and the 3D scene within the window.
        """
        r = self.window.content_rect
        left_w = 440
        scene_w = max(1, r.width - left_w)
        self.panel.frame = gui.Rect(r.x, r.y, left_w, r.height)
        self.scene.frame = gui.Rect(r.x + left_w, r.y, scene_w, r.height)

    def _has_points(self, pc):
        """
        Returns True if an object is a non-empty PointCloud.
        """
        return pc is not None and isinstance(pc, o3d.geometry.PointCloud) and len(pc.points) > 0

    def _clear_3d_labels(self):
        """
        Clears 3D labels if the renderer supports it.
        """
        if hasattr(self.scene.scene, "clear_3d_labels"):
            try:
                self.scene.scene.clear_3d_labels()
            except Exception:
                pass

    def _add_3d_label(self, pos, text):
        """
        Adds a 3D label at position `pos` with content `text`, if supported.
        """
        if hasattr(self.scene.scene, "add_3d_label"):
            try:
                self.scene.scene.add_3d_label(pos, text)
            except Exception:
                pass

    def show_geometry(self, geometries):
        """
        Clears and re-adds geometries to the scene; updates camera & labels.

        Parameters
        ----------
        geometries : o3d geometry or list
            Geometry(ies) to show.
        """
        # Clear geometries
        self.scene.scene.clear_geometry()
        # Clear labels; will re-add if enabled
        self._clear_3d_labels()

        if not isinstance(geometries, list):
            geometries = [geometries]
        merged = o3d.geometry.PointCloud()
        for i, geom in enumerate(geometries):
            if isinstance(geom, o3d.geometry.PointCloud):
                mat = self.mat_points; merged += geom
            elif isinstance(geom, o3d.geometry.LineSet):
                mat = self.mat_lines
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                mat = self.mat_mesh
            else:
                mat = self.mat_points
            self.scene.scene.add_geometry(f"g{i}", geom, mat)

        # Add centroid spheres if enabled
        k0 = len(geometries)
        for j, mesh in enumerate(self.centroid_spheres):
            self.scene.scene.add_geometry(f"centroid_{j}", mesh, self.mat_mesh)

        # Re-add 3D labels if enabled
        if self.centroid_labels_enabled and self.centroid_labels_data:
            for pos, text in self.centroid_labels_data:
                self._add_3d_label(pos, text)

        if len(merged.points) > 0:
            bounds = merged.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60, bounds, bounds.get_center())

    # ---------- Utility ----------

    def _get_selected_text(self, combo: gui.Combobox, items):
        """
        Returns the selected string from a Combobox; falls back to index lookup.
        """
        if hasattr(combo, "selected_text"):
            try:
                return combo.selected_text
            except Exception:
                pass
        idx = combo.selected_index if hasattr(combo, "selected_index") else 0
        if idx < 0 or idx >= len(items):
            idx = 0
        return items[idx]

    def _estimate_marker_radius(self):
        """
        Heuristically estimates a nice radius for centroid spheres from scene extent.
        """
        # Use current clusters if present, else original cloud
        if self.clusters:
            merged = o3d.geometry.PointCloud()
            for cl in self.clusters:
                merged += cl
            bounds = merged.get_axis_aligned_bounding_box()
        else:
            bounds = self.original_pcd.get_axis_aligned_bounding_box()
        extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        r = max(0.004 * extent, 0.001)  # ~0.4% of scene size, min 1mm (meters units)
        return float(r)

    def _make_centroid_sphere(self, center, radius, color=(0.1, 0.1, 0.1)):
        """
        Creates a small sphere mesh at `center` with given `radius` and `color`.
        """
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
        mesh.translate(center)
        return mesh

    # ---------- Actions ----------

    def apply_z_filter(self):
        """
        Filters points by Z range using the two sliders; shows filtered or original cloud.
        """
        z0 = self.slider_min.double_value; z1 = self.slider_max.double_value
        if z0 > z1: z0, z1 = z1, z0
        Z = np.asarray(self.original_pcd.points)[:, 2]
        mask = (Z >= z0) & (Z <= z1)
        idx = np.nonzero(mask)[0].tolist()
        self.filtered_pcd = self.original_pcd.select_by_index(idx)
        self.show_geometry(self.filtered_pcd if self._has_points(self.filtered_pcd) else self.original_pcd)

    def preprocess(self):
        """
        Runs voxel downsample + statistical outlier removal on the filtered cloud.
        """
        if not self._has_points(self.filtered_pcd): return
        voxel = self.voxel_slider.double_value; std = self.std_slider.double_value
        self.processed_pcd = preprocess_point_cloud(self.filtered_pcd, voxel_size=voxel, std_ratio=std)
        self.show_geometry(self.processed_pcd)

    def detect_planes(self):
        """
        Detects multiple planes with RANSAC from the processed cloud; updates sliders & view.
        """
        if not self._has_points(self.processed_pcd): return
        dist = self.dist_thresh_slider.double_value
        n = self.ransac_n_slider.int_value
        iterations = self.ransac_iter_slider.int_value
        self.planes_cluster, self.planes_display, self.remaining_cloud = detect_multiple_planes(
            self.processed_pcd, dist, n, iterations
        )
        max_plane_idx = max(0, len(self.planes_display) - 1)
        self.plane_index_slider.set_limits(0, max_plane_idx); self.plane_index_slider.int_value = 0
        self.selected_plane_idx = 0
        display = list(self.planes_display)
        if self._has_points(self.remaining_cloud):
            grey = o3d.geometry.PointCloud(self.remaining_cloud); grey.paint_uniform_color([0.7, 0.7, 0.7]); display.append(grey)
        self.show_geometry(display)
        # Update report buffer (not shown, used for saving)
        self.last_report_lines = ["Color Grouping Report", "Planes detected. Cluster and then group to generate a report."]

    def show_selected_plane(self):
        """
        Displays only the plane selected by the plane index slider (+ optional remaining cloud).
        """
        if not self.planes_display: return
        idx = max(0, min(self.plane_index_slider.int_value, len(self.planes_display) - 1))
        self.selected_plane_idx = idx
        display = [self.planes_display[idx]]
        if self._has_points(self.remaining_cloud):
            grey = o3d.geometry.PointCloud(self.remaining_cloud); grey.paint_uniform_color([0.85, 0.85, 0.85]); display.append(grey)
        self.show_geometry(display)

    def _capture_cluster_display_colors(self):
        """
        Stores the current painted color of each display cluster for later reset.
        """
        self.cluster_disp_colors = []
        for cl in self.clusters:
            if len(cl.colors) > 0:
                self.cluster_disp_colors.append(np.asarray(cl.colors)[0].copy())
            else:
                self.cluster_disp_colors.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    def _recompute_hull_centroids_for_all(self):
        """
        Recomputes and stores 2D-hull-based 3D centroids for all display clusters.
        """
        self.cluster_hull_centroids = []
        for cl in self.clusters:
            _, c3 = get_hull_mask_and_centroid3d(cl)
            self.cluster_hull_centroids.append(c3)

    def cluster_planes_all(self):
        """
        Runs DBSCAN clustering on every detected plane; populates display & original clusters.
        """
        if not self.planes_cluster: return
        eps = self.eps_slider.double_value
        min_samples = self.min_samples_slider.int_value
        color_weight = self.color_slider.double_value
        pos_weight = self.pos_slider.double_value
        self.clusters = []; self.cluster_originals = []; self.clusters_by_plane.clear()
        for i, plane in enumerate(self.planes_cluster):
            clusters_disp, clusters_orig = cluster_colored_plane(plane, eps, min_samples, color_weight, pos_weight)
            self.clusters_by_plane[i] = clusters_disp
            self.clusters.extend(clusters_disp); self.cluster_originals.extend(clusters_orig)
        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()
        max_cluster_idx = max(0, len(self.clusters) - 1)
        self.cluster_index_slider.set_limits(0, max_cluster_idx); self.cluster_index_slider.int_value = 0
        self.selected_cluster_idx = 0
        display = list(self.clusters)
        if self._has_points(self.remaining_cloud):
            grey = o3d.geometry.PointCloud(self.remaining_cloud); grey.paint_uniform_color([0.85, 0.85, 0.85]); display.append(grey)
        self.show_geometry(display)
        self.last_report_lines = ["Color Grouping Report", "Clusters ready. Run 'Group Clusters by Color' then Save Report."]

    def cluster_selected_plane(self):
        """
        Runs DBSCAN clustering only on the currently selected plane.
        """
        if not self.planes_cluster: return
        idx = max(0, min(self.plane_index_slider.int_value, len(self.planes_cluster) - 1))
        self.selected_plane_idx = idx
        eps = self.eps_slider.double_value
        min_samples = self.min_samples_slider.int_value
        color_weight = self.color_slider.double_value
        pos_weight = self.pos_slider.double_value
        clusters_disp, clusters_orig = cluster_colored_plane(self.planes_cluster[idx], eps, min_samples, color_weight, pos_weight)
        self.clusters_by_plane[idx] = clusters_disp
        self.clusters = clusters_disp; self.cluster_originals = clusters_orig
        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()
        max_idx = max(0, len(clusters_disp) - 1)
        self.cluster_index_slider.set_limits(0, max_idx); self.cluster_index_slider.int_value = 0
        self.selected_cluster_idx = 0
        display = list(clusters_disp)
        self.show_geometry(display)
        self.last_report_lines = ["Color Grouping Report", "Clusters ready. Run 'Group Clusters by Color' then Save Report."]

    # ---- HULLS: 3D ----
    def compute_hull_selected(self):
        """
        Computes a 3D convex hull (as LineSet) for the selected cluster and shows it.
        """
        if not self.clusters: print("No clusters to hull."); return
        cidx = max(0, min(self.cluster_index_slider.int_value, len(self.clusters) - 1))
        self.selected_cluster_idx = cidx
        cluster = self.clusters[cidx]
        hull, _ = cluster.compute_convex_hull()
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull); lines.paint_uniform_color([0, 0, 0])
        self.hull_selected = lines
        display = list(self.clusters) + [lines]
        self.show_geometry(display)

    def compute_convex_hulls_all(self):
        """
        Computes 3D convex hulls for all clusters and overlays them.
        """
        if not self.clusters: print("No clusters to hull."); return
        self.hull_lines = []
        for cluster in self.clusters:
            hull, _ = cluster.compute_convex_hull()
            lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull); lines.paint_uniform_color([0, 0, 0])
            self.hull_lines.append(lines)
        display = list(self.clusters) + self.hull_lines
        self.show_geometry(display)

    # ---- HULLS: 2D (projected) ----
    def compute_hull_selected_2d(self):
        """
        Computes the 2D in-plane hull for the selected cluster and displays it.
        """
        if not self.clusters: print("No clusters to hull."); return
        cidx = max(0, min(self.cluster_index_slider.int_value, len(self.clusters) - 1))
        self.selected_cluster_idx = cidx
        cluster = self.clusters[cidx]
        ls = create_2d_hull_lineset_from_cluster(cluster)
        if ls is None:
            print("Could not compute 2D hull (cluster too small)."); return
        self.hull_selected_2d = ls
        _, c3 = get_hull_mask_and_centroid3d(cluster)
        if len(self.cluster_hull_centroids) == len(self.clusters):
            self.cluster_hull_centroids[cidx] = c3
        display = list(self.clusters) + [ls]
        self.show_geometry(display)

    def compute_convex_hulls_all_2d(self):
        """
        Computes the 2D in-plane hulls for all clusters and overlays them.
        """
        if not self.clusters: print("No clusters to hull."); return
        self.hull_lines_2d = []; self._recompute_hull_centroids_for_all()
        for cluster in self.clusters:
            ls = create_2d_hull_lineset_from_cluster(cluster)
            if ls is not None:
                self.hull_lines_2d.append(ls)
        display = list(self.clusters) + self.hull_lines_2d
        self.show_geometry(display)

    # ---- Reset & Grouping ----
    def reset_display_colors(self):
        """
        Restores each display cluster to the color it had before grouping.
        """
        if not self.clusters or not self.cluster_disp_colors:
            print("Nothing to reset.")
            self.last_report_lines = ["Color Grouping Report", "Nothing to reset."]
            return
        for cl, col in zip(self.clusters, self.cluster_disp_colors):
            cl.paint_uniform_color(col.tolist())
        display = list(self.clusters)
        if self.hull_lines: display += self.hull_lines
        if self.hull_lines_2d: display += self.hull_lines_2d
        # keep centroids if enabled
        self.show_geometry(display)
        self.last_report_lines = ["Color Grouping Report", "Display colors reset."]

    def group_clusters_color_action(self):
        """
        Groups clusters by color feature; recolors display clusters by group mean RGB.
        """
        if not self.cluster_originals or not self.clusters:
            print("No clusters available. Run clustering first.")
            self.last_report_lines = ["Color Grouping Report", "No clusters available. Run clustering first."]
            return
        mode = self._get_selected_text(self.group_mode_combo, self.group_mode_items)
        eps_z = self.group_eps_slider.double_value
        min_size = self.group_min_slider.int_value
        use_hull_mask = self.group_use_hull_checkbox.checked
        centroids = []
        labels, group_colors, report_lines = group_clusters_by_color(
            self.cluster_originals, mode, eps_z, min_size,
            use_hull_mask=use_hull_mask, centroids_out=centroids
        )
        if labels.size == 0:
            print("Color grouping produced no labels.")
            self.last_report_lines = ["Color Grouping Report", "Color grouping produced no labels."]
            return

        # recolor display clusters by group color
        for i, cl in enumerate(self.clusters):
            g = labels[i]
            if g == -1:
                continue
            rgb = group_colors.get(g, np.array([random.random(), random.random(), random.random()]))
            cl.paint_uniform_color(rgb.tolist())

        # ---- NEW: store grouping for JSON export ----
        self.color_group_labels = labels  # np.ndarray
        # convert mean colors to plain lists for JSON
        self.color_group_mean_rgb = {int(k): group_colors[k].tolist() for k in group_colors}

        # build and store a compact summary (mode/eps/min_size/hull_mask + groups)
        groups_summary = []
        for g in sorted(set(labels)):
            if g == -1:
                continue
            members = np.where(labels == g)[0].tolist()
            groups_summary.append({
                "group_id": int(g),
                "mean_rgb": self.color_group_mean_rgb[int(g)],
                "members": [int(x) for x in members]
            })
        self.last_grouping_meta = {
            "mode": mode,
            "eps_z": float(eps_z),
            "min_size": int(min_size),
            "use_hull_mask": bool(use_hull_mask),
            "labels": labels.tolist(),
            "groups": groups_summary
        }

        self.cluster_hull_centroids = centroids  # keep latest centroids (also used for labels)
        display = list(self.clusters)
        if self.hull_lines: display += self.hull_lines
        if self.hull_lines_2d: display += self.hull_lines_2d
        self.show_geometry(display)
        self.last_report_lines = report_lines

    # ---- Centroid visualization ----
    def show_centroids(self):
        """
        Shows small spheres and optional labels at per-cluster 2D-hull centroids.
        """
        if not self.clusters:
            print("No clusters available. Run clustering first.")
            return
        # recompute centroids (2D hull)
        self._recompute_hull_centroids_for_all()
        r = self._estimate_marker_radius()
        self.centroid_spheres = []
        self.centroid_labels_data = []
        for i, c3 in enumerate(self.cluster_hull_centroids):
            if c3 is None or np.any(np.isnan(c3)):
                continue
            mesh = self._make_centroid_sphere(c3, r, color=(1, 0, 0))
            self.centroid_spheres.append(mesh)
            txt = f"c{i}: ({c3[0]:.3f}, {c3[1]:.3f}, {c3[2]:.3f})"
            self.centroid_labels_data.append((c3, txt))
        self.centroid_labels_enabled = True
        display = list(self.clusters)
        if self.hull_lines: display += self.hull_lines
        if self.hull_lines_2d: display += self.hull_lines_2d
        self.show_geometry(display)

    def hide_centroids(self):
        """
        Hides centroid spheres and labels.
        """
        self.centroid_spheres = []
        self.centroid_labels_enabled = False
        self.centroid_labels_data = []
        display = list(self.clusters)
        if self.hull_lines: display += self.hull_lines
        if self.hull_lines_2d: display += self.hull_lines_2d
        self.show_geometry(display)

    # ---- Save report ----
    def save_report_to_txt(self):
        """
        Writes the current color-grouping report lines to a timestamped .txt file.
        """
        lines = self.last_report_lines if self.last_report_lines else ["(empty report)"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"Processing Report_{ts}.txt"
        path = os.path.join(self.export_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Saved report: {path}")
        except Exception as e:
            print(f"Failed to save report: {e}")

    # ---- Export Selected Cluster (.json) ----
    def export_selected_cluster_json(self):
        """
        Exports JSON for the currently selected cluster into `self.export_dir`.

        The JSON includes:
        - plane frame (origin, u, v, n)
        - 3D polygon of the in-plane hull
        - hull centroid in 3D
        - color grouping info for this cluster (group_id and group mean RGB if available)
        """
        if not self.clusters:
            print("No clusters to export. Run clustering first.")
            return
        cidx = max(0, min(self.cluster_index_slider.int_value, len(self.clusters) - 1))
        # use original (unpainted) cluster if available
        orig_cluster = self.cluster_originals[cidx] if cidx < len(self.cluster_originals) else self.clusters[cidx]

        # plane frame, hull, centroid
        poly2, poly3, origin, u, v, n, centroid3d = compute_hull_poly2_poly3(orig_cluster)

        # ---- color grouping per-cluster mapping ----
        cg = None
        if isinstance(self.color_group_labels, np.ndarray) and len(self.color_group_labels) == len(self.clusters):
            g_id = int(self.color_group_labels[cidx])
            if g_id != -1:
                cg = {
                    "group_id": g_id,
                    "group_mean_rgb": self.color_group_mean_rgb.get(g_id, None)
                }
            else:
                cg = {"group_id": -1}  # noise / unassigned

        payload = {
            "units": "meters",
            "timestamp": datetime.now().isoformat(),
            "plane_index": int(self.selected_plane_idx),
            "cluster_index": int(cidx),
            "plane_frame": {
                "origin": origin.tolist(),
                "u": u.tolist(),
                "v": v.tolist(),
                "n": n.tolist()
            },
            "hull_3d": poly3.tolist() if poly3 is not None else [],
            "centroid_3d": centroid3d.tolist()
        }

        if cg is not None:
            payload["color_group"] = cg

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"cluster_plane{payload['plane_index']}_cluster{cidx}_{ts}.json"
        path = os.path.join(self.export_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Exported selected cluster JSON: {path}")
        except Exception as e:
            print(f"Failed to export JSON: {e}")

    # ---- Export ALL Clusters (.json) ----
    def export_all_clusters_json(self):
        """
        Exports JSON describing all current clusters into `self.export_dir`.
        Includes per-cluster color grouping mapping and a top-level grouping summary if available.
        """
        if not self.clusters:
            print("No clusters to export. Run clustering first.")
            return

        have_groups = isinstance(self.color_group_labels, np.ndarray) and len(self.color_group_labels) == len(self.clusters)

        entries = []
        for cidx, disp_cluster in enumerate(self.clusters):
            orig_cluster = self.cluster_originals[cidx] if cidx < len(self.cluster_originals) else disp_cluster
            poly2, poly3, origin, u, v, n, centroid3d = compute_hull_poly2_poly3(orig_cluster)

            color_group = None
            if have_groups:
                g_id = int(self.color_group_labels[cidx])
                if g_id != -1:
                    color_group = {
                        "group_id": g_id,
                        "group_mean_rgb": self.color_group_mean_rgb.get(g_id, None)
                    }
                else:
                    color_group = {"group_id": -1}

            entry = {
                "plane_index": int(self.selected_plane_idx),   # current slider value context
                "cluster_index": int(cidx),
                "plane_frame": {
                    "origin": origin.tolist(),
                    "u": u.tolist(),
                    "v": v.tolist(),
                    "n": n.tolist()
                },
                "hull_3d": poly3.tolist() if poly3 is not None else [],
                "centroid_3d": centroid3d.tolist()
            }
            if color_group is not None:
                entry["color_group"] = color_group

            entries.append(entry)

        payload = {
            "units": "meters",
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(entries),
            "clusters": entries
        }

        # ---- add overall color grouping summary if available ----
        # if self.last_grouping_meta is not None:
        #     payload["color_grouping"] = self.last_grouping_meta

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"ALL_CLUSTERS_{ts}.json"
        path = os.path.join(self.export_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Exported ALL clusters JSON: {path}")
        except Exception as e:
            print(f"Failed to export JSON: {e}")

# ---------- Main ----------

def main():
    """
    Entry point: loads point cloud, converts mm->m, launches the GUI app.
    """
    gui.Application.instance.initialize()
    
    pcd = o3d.io.read_point_cloud(os.path.join(import_directory, filename))
    if len(pcd.points) == 0:
        print("Point cloud is empty."); return
    # mm -> m
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 1000.0)
    SegmentationApp(pcd)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
import open3d as o3d
import numpy as np
import os, re, copy
from glob import glob

# ─────────────── CONFIGURATION ───────────────
pointcloud_dir = r"F:\\07. IAAC_Internship\\00. PROJECT 01_deco2_robotic_mosaic\\ICP\\ICP_1st Trial\\test05-1\\Initial point clouds"
transform_dir  = r"F:\\07. IAAC_Internship\\00. PROJECT 01_deco2_robotic_mosaic\\ICP\\ICP_1st Trial\\test05-1\\Transformation matrix_3 capture"
output_path    = r"F:\\07. IAAC_Internship\\00. PROJECT 01_deco2_robotic_mosaic\\ICP\\ICP_1st Trial\\test05-1\\Merged point clouds\\merged01.ply"

reference_index = 1  # <-- anchor scan index

VOXEL_SIZE_MM = 0.5
NORMAL_RADIUS_MM = 1.0
ICP_THRESHOLDS = [10.0, 5.0, 1.0]  # mm
# Choose pipeline: "ref_cam" (original) or "base" (pre-transform to robot base then ICP)
PIPELINE_FRAME = "base"
# ─────────────────────────────────────────────


class PointCloudMerger:
    def __init__(self, pcd_dir, tf_dir, reference_index,
                 voxel_size_mm=0.5, normal_radius_mm=1.0, icp_thresholds_mm=(10.0, 5.0, 1.0),
                 frame_mode="ref_cam"):
        self.pcd_dir = pcd_dir
        self.tf_dir = tf_dir
        self.reference_index = reference_index
        self.voxel = voxel_size_mm
        self.normal_radius = normal_radius_mm
        self.icp_thresholds = icp_thresholds_mm
        assert frame_mode in ("ref_cam", "base")
        self.frame_mode = frame_mode

        self.pcd_dict = {}
        self.tf_dict = {}
        self.T_ref = np.eye(4)

    # ---------- utils ----------
    @staticmethod
    def load_transform_matrix(txt_path: str) -> np.ndarray:
        with open(txt_path, "r") as f:
            rows = [line.strip("[]\n ").split(",") for line in f.readlines()]
        return np.array([[float(x) for x in row] for row in rows])

    @staticmethod
    def get_index_from_pcd(file):
        match = re.search(r"point_cloud_(\d+)", os.path.basename(file))
        return int(match.group(1)) if match else None

    @staticmethod
    def get_index_from_tf(file):
        match = re.search(r"T_base_cam_pose(\d+)", os.path.basename(file))
        return int(match.group(1)) if match else None

    @staticmethod
    def show_pair(src, tgt, T=np.eye(4), title=""):
        s, t = copy.deepcopy(src), copy.deepcopy(tgt)
        s.paint_uniform_color([1.0, 0.6, 0.0]); t.paint_uniform_color([0.0, 0.65, 0.93])
        s.transform(T)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(s); vis.add_geometry(t)
        vis.get_view_control().set_zoom(0.8)
        vis.run(); vis.destroy_window()

    @staticmethod
    def run_icp(src, tgt, threshold_mm, init_T, estimation):
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold_mm, init_T, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        return result.transformation

    def process_point_cloud(self, pcd_path, tf_path, pretransform_to_base=False):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(self.voxel)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        T_base_cam = self.load_transform_matrix(tf_path)
        if pretransform_to_base:
            # T_base_cam maps points from camera to base: p_base = T_base_cam * p_cam
            pcd.transform(T_base_cam)
        return pcd, T_base_cam

    # ---------- pipelines ----------
    def merge_ref_cam_pipeline(self):
        """Original behavior: align in reference CAMERA frame; return merged (camera frame) + T_ref."""
        all_pcd_files = glob(os.path.join(self.pcd_dir, "point_cloud_*.ply"))
        all_tf_files  = glob(os.path.join(self.tf_dir, "T_base_cam_pose*.txt"))
        self.pcd_dict = {self.get_index_from_pcd(f): f for f in all_pcd_files if self.get_index_from_pcd(f) is not None}
        self.tf_dict  = {self.get_index_from_tf(f): f for f in all_tf_files  if self.get_index_from_tf(f)  is not None}

        print("Found point clouds:", sorted(self.pcd_dict.keys()))
        print("Found transforms:  ", sorted(self.tf_dict.keys()))
        if self.reference_index not in self.pcd_dict or self.reference_index not in self.tf_dict:
            raise FileNotFoundError(f"Reference index {self.reference_index} not found in files.")

        ref_pcd, T_ref = self.process_point_cloud(self.pcd_dict[self.reference_index],
                                                  self.tf_dict[self.reference_index],
                                                  pretransform_to_base=False)
        self.T_ref = T_ref
        merged = copy.deepcopy(ref_pcd)

        for i in sorted(self.pcd_dict.keys()):
            if i == self.reference_index:
                continue
            if i not in self.tf_dict:
                print(f"[!] Missing transform for cloud {i}, skipping."); continue

            print(f"\n[•] Aligning point_cloud_{i:02d} to reference point_cloud_{self.reference_index:02d}...")
            src_pcd, T_i = self.process_point_cloud(self.pcd_dict[i], self.tf_dict[i], pretransform_to_base=False)

            # Same math as before: bring src into reference CAMERA frame
            T_init = np.linalg.inv(self.T_ref) @ T_i
            T_A = self.run_icp(src_pcd, merged, self.icp_thresholds[0], T_init,
                               o3d.pipelines.registration.TransformationEstimationPointToPoint())
            T_B = self.run_icp(src_pcd, merged, self.icp_thresholds[1], T_A,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_C = self.run_icp(src_pcd, merged, self.icp_thresholds[2], T_B,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())

            src_pcd.transform(T_C)
            self.show_pair(src_pcd, merged, np.eye(4), f"Merged with point_cloud_{i:02d}")
            merged += src_pcd
            merged = merged.voxel_down_sample(self.voxel)

        return merged, self.T_ref  # merged in reference CAMERA frame

    def merge_base_pipeline(self):
        """Proposed behavior: pre-transform each cloud into BASE using T_base_cam, then ICP in BASE."""
        all_pcd_files = glob(os.path.join(self.pcd_dir, "point_cloud_*.ply"))
        all_tf_files  = glob(os.path.join(self.tf_dir, "T_base_cam_pose*.txt"))
        self.pcd_dict = {self.get_index_from_pcd(f): f for f in all_pcd_files if self.get_index_from_pcd(f) is not None}
        self.tf_dict  = {self.get_index_from_tf(f): f for f in all_tf_files  if self.get_index_from_tf(f)  is not None}

        print("Found point clouds:", sorted(self.pcd_dict.keys()))
        print("Found transforms:  ", sorted(self.tf_dict.keys()))
        if self.reference_index not in self.pcd_dict or self.reference_index not in self.tf_dict:
            raise FileNotFoundError(f"Reference index {self.reference_index} not found in files.")

        # Reference cloud, already in BASE
        ref_pcd_base, T_ref = self.process_point_cloud(self.pcd_dict[self.reference_index],
                                                       self.tf_dict[self.reference_index],
                                                       pretransform_to_base=True)
        self.T_ref = T_ref  # not used for inverses in this pipeline, retained for logging if needed
        merged = copy.deepcopy(ref_pcd_base)

        for i in sorted(self.pcd_dict.keys()):
            if i == self.reference_index:
                continue
            if i not in self.tf_dict:
                print(f"[!] Missing transform for cloud {i}, skipping."); continue

            print(f"\n[•] Aligning (BASE) point_cloud_{i:02d} to merged...")
            # Source cloud, already in BASE
            src_pcd_base, _ = self.process_point_cloud(self.pcd_dict[i], self.tf_dict[i],
                                                       pretransform_to_base=True)

            # Since both are in BASE, a good init is the identity
            T_A = self.run_icp(src_pcd_base, merged, self.icp_thresholds[0], np.eye(4),
                               o3d.pipelines.registration.TransformationEstimationPointToPoint())
            T_B = self.run_icp(src_pcd_base, merged, self.icp_thresholds[1], T_A,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_C = self.run_icp(src_pcd_base, merged, self.icp_thresholds[2], T_B,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())

            src_pcd_base.transform(T_C)
            self.show_pair(src_pcd_base, merged, np.eye(4), f"Merged (BASE) with point_cloud_{i:02d}")
            merged += src_pcd_base
            merged = merged.voxel_down_sample(self.voxel)

        # Result is already in BASE; no final transform needed
        return merged

    def run(self, output_path):
        if self.frame_mode == "ref_cam":
            merged_cam, T_ref = self.merge_ref_cam_pipeline()
            # If you still want the final in BASE, uncomment the line below:
            # merged_cam.transform(np.linalg.inv(T_ref))
            o3d.io.write_point_cloud(output_path, merged_cam)
            print(f"\nSaved merged cloud (reference CAMERA frame) → {output_path}")
            o3d.visualization.draw_geometries([merged_cam], window_name="Merged (Ref Camera)")
        else:
            merged_base = self.merge_base_pipeline()
            o3d.io.write_point_cloud(output_path, merged_base)
            print(f"\nSaved merged cloud (ROBOT BASE frame) → {output_path}")
            o3d.visualization.draw_geometries([merged_base], window_name="Merged (Robot Base)")


def main():
    merger = PointCloudMerger(
        pcd_dir=pointcloud_dir,
        tf_dir=transform_dir,
        reference_index=reference_index,
        voxel_size_mm=VOXEL_SIZE_MM,
        normal_radius_mm=NORMAL_RADIUS_MM,
        icp_thresholds_mm=ICP_THRESHOLDS,
        frame_mode=PIPELINE_FRAME  # "base" uses your proposed approach
    )
    merger.run(output_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Connect to a Mech-Eye Industrial 3D Camera, capture, and save a textured point cloud
to a specified directory.
"""

import os
from datetime import datetime

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import print_camera_info, show_error


class CaptureTexturedPointCloud:
    def __init__(self):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()

        # === Configure output directory here ===
        self.output_dir = r"F:\\07. IAAC_Internship\\00. PROJECT 01_deco2_robotic_mosaic\\ICP\\ICP_1st Trial\\test05-1\\Initial point clouds"  # Change this path as needed
        os.makedirs(self.output_dir, exist_ok=True)

    def discover_cameras(self):
        print("Discovering all available cameras...")
        camera_infos = Camera.discover_cameras()
        if len(camera_infos) == 0:
            print("No cameras found.")
            return []

        # Display available cameras
        for i, info in enumerate(camera_infos):
            print(f"\nCamera index : {i}")
            print_camera_info(info)
        return camera_infos

    def choose_camera_index(self, camera_infos):
        while True:
            index_str = input(
                f"\nEnter the index of the camera to connect (0–{len(camera_infos) - 1}, default 0): "
            ).strip()
            if index_str == "":
                return 0
            if index_str.isdigit() and 0 <= int(index_str) < len(camera_infos):
                return int(index_str)
            print("Invalid input. Please enter a valid index.")

    def capture_textured_point_cloud(self):
        """Capture a textured point cloud and save it to a timestamped file."""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"point_cloud_01.ply")

        print("\nCapturing 2D + 3D frame...")
        status = self.camera.capture_2d_and_3d(self.frame_all_2d_3d)
        if not status.is_ok():
            show_error(status)
            return False

        print(f"Saving textured point cloud to: {output_file}")
        status = self.frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY, output_file)
        if not status.is_ok():
            show_error(status)
            return False

        print(f"Textured point cloud saved as: {output_file}")
        return True

    def main(self):
        # Discover available cameras
        camera_infos = self.discover_cameras()
        if not camera_infos:
            return

        # Choose camera
        cam_index = self.choose_camera_index(camera_infos)

        # Connect to selected camera
        print(f"\nConnecting to camera index {cam_index}...")
        status = self.camera.connect(camera_infos[cam_index])
        if not status.is_ok():
            show_error(status)
            return

        print("Connected to the camera successfully.")

        # Capture and save textured point cloud
        self.capture_textured_point_cloud()

        # Disconnect
        self.camera.disconnect()
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    app = CaptureTexturedPointCloud()
    app.main()

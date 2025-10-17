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

# import the session helpers (directory management only)
from helpers.session_manager import load_session, init_session


class CaptureTexturedPointCloud:
    def __init__(self):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()

        # === Directory now resolved per session (Initial point clouds) ===
        # We set this later in main() after choosing/creating the session.
        self.output_dir = None

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
        """Capture a textured point cloud and save it with an index-based filename."""
        # Ask for index and build zero-padded filename
        while True:
            idx_str = input("\nEnter point cloud index [1..99] (default 1): ").strip()
            if idx_str == "":
                idx = 1
                break
            if idx_str.isdigit() and 1 <= int(idx_str) <= 99:
                idx = int(idx_str)
                break
            print("Invalid input. Please enter a number between 1 and 99.")

        filename = f"point_cloud_{idx:02d}.ply"
        output_file = os.path.join(self.output_dir, filename)

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
        # >>> Added: pick or create a session, then set output_dir to its "Initial point clouds"
        session_name = input("Session name (leave blank to reuse last or create 'Test 01'): ").strip()
        try:
            paths = load_session(".", session_name or None)
        except FileNotFoundError:
            # If no session exists, create one (named by user or default)
            paths = init_session(".", session_name or "Test 01")
            print(f"[init] created session: {paths.session_name}")

        # Match your original intent: save into "Initial point clouds"
        self.output_dir = str(paths.initial_point_clouds)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[capture] Output directory set to: {self.output_dir}")

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
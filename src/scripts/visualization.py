import open3d as o3d
import numpy as np
import sys

arr = np.load(f"./datasets/Synth20/pointcloud1/{sys.argv[1]}.npy")

print(arr)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(arr)

o3d.draw_geometries([pcd])
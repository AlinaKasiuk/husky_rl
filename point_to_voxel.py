# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:08:22 2023

@author: Alina
"""
import numpy as np
import open3d as o3d


def observation(filename):
    point_cloud = np.loadtxt('Points.txt', delimiter=' ')
    point_cloud = point_cloud[((np.square(point_cloud[:, 0]) + np.square(point_cloud[:, 1])) < 400) &
                              (np.absolute(point_cloud[:, 2]) < 20)]
    points = point_cloud/500
    return points

points = observation('Points.txt')
new_points = observation('Points\Point0_2.txt')

points = np.append(points,new_points, axis=0)



pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
#pcd.colors = o3d.utility.Vector3dVector(colors)



# Create a voxel grid from the point cloud with a voxel_size of 0.01
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.0001)

# Initialize a visualizer object
vis = o3d.visualization.Visualizer()
# Create a window, name it and scale it
vis.create_window(window_name='Bunny Visualize', width=800, height=600)

# Add the voxel grid to the visualizer
vis.add_geometry(voxel_grid)

# We run the visualizater
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()

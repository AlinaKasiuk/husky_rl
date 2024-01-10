# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:08:22 2023

@author: Alina
"""
import numpy as np
import open3d as o3d


def observation(filename):
    point_cloud = np.loadtxt(filename, delimiter=' ')
    point_cloud = point_cloud[((np.square(point_cloud[:, 0]) + np.square(point_cloud[:, 1])) < 400) &
                              (np.absolute(point_cloud[:, 2]) < 20)]
    colors = np.asarray( [np.array([95,158,160]) + np.array([11,-68,45])*int(a) for a in (point_cloud[:,3]==4284456608)])/255
    points = point_cloud[:,:3]/500
    return points, colors

def position(filename):
    pose =  np.loadtxt(filename, delimiter=' ')
    return pose[:3], pose[3:6]

points, colors = observation('Exp2/Points/Point0_0.txt')
pose = position('Exp2/Poses/Pose0_0.txt')

for i in range(1,100):
    new_points,  new_colors = observation('Exp2/Points/Point0_'+str(i)+'.txt')
    new_pose, new_orient = position('Exp2/Poses/Pose0_'+str(i)+'.txt')
    new_points += new_pose/500
    points = np.append(points,new_points, axis=0)
    colors = np.append(colors,new_colors, axis=0)

print(points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)



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

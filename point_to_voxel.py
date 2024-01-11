# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:08:22 2023

@author: Alina
"""
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
     

                            
    return rot_matrix

def observation(filename):
    point_cloud = np.loadtxt(filename, delimiter=' ')
    point_cloud = point_cloud[((np.square(point_cloud[:, 0]) + np.square(point_cloud[:, 1])) < 400) &
                              (np.absolute(point_cloud[:, 2]) < 20)]
    colors = np.asarray( [np.array([95,158,160]) + np.array([11,-68,45])*int(a) for a in (point_cloud[:,3]==4284456608)])/255
    points = point_cloud[:,:3]/500
    return points, colors

def position(filename):
    pose =  np.loadtxt(filename, delimiter=' ')
    return pose[:3], pose[3:]

points, colors = observation('Exp2/Points/Point0_0.txt')
pose, orient = position('Exp2/Poses/Pose0_0.txt')
#R = Rotation.from_quat(orient).as_matrix()
#pose = np.dot(pose, R)

for i in range(1,100):
    new_points,  new_colors = observation('Exp2/Points/Point0_'+str(i)+'.txt')
    new_pose, new_orient = position('Exp2/Poses/Pose0_'+str(i)+'.txt')
    
    
    new_orient = np.append(new_orient[3], new_orient[:3])
    R = quaternion_rotation_matrix(new_orient)
    
    R_1 = Rotation.from_quat(new_orient)
    
    new_points = np.dot(new_points + new_pose/500 , np.linalg.inv(R))
    
    # Add orientation transformation
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

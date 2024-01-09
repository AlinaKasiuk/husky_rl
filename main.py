#Una clase que se subscribe al topico del ouster que esa nube la
#haga pasar por la red y que publique la nube en un nuevo topico llamado
#traversability con colores para visualizar en rviz

import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.abspath("/home/arvc/venv/MinkowskiEngine/examples"))
from minkunet import MinkUNet34C
import time
sys.path.append(os.path.abspath("/home/arvc/venv/MinkowskiEngine/examples"))
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import struct

class traversability_ROS():
    def __init__(self):
        self.device = torch.device('cpu')
        self.root="/home/arvc/venv/scripts/BestModel3_th_0.23192343544214963voxel_size0.2_0.9207645875913779.pth"
        self.model = MinkUNet34C(1, 1).to(self.device)
        self.model.load_state_dict(torch.load(self.root, map_location=torch.device('cpu')))
        self.criterion = nn.BCELoss()
        self.optimizer = SGD(self.model.parameters(), lr=1e-1)
        self.threshold=0.23192343544214963
        self.voxel_size=0.2
        self.counter=0
        self.pub = rospy.Publisher("trav_analysis", PointCloud2, queue_size=2)

    def listener(self):
        self.sub=rospy.Subscriber("/os1/pointCloud", PointCloud2, self.callback)

    def callback(self, data):
        start = time.time()
        field_names = [field.name for field in data.fields]
        # field_names = ['x', 'y', 'z']
        if self.counter%10==0:
            points = list(pc2.read_points(data, skip_nans=True, field_names=field_names))

            if len(points) == 0:
                print("Converting an empty cloud")
                return None
            else:#cambiar el calculo de normales por un vector de unos	
                pcd_array = np.asarray(points)
                pointcloud = o3d.geometry.PointCloud()
                pointcloud.points= o3d.utility.Vector3dVector(pcd_array[:, 0:3])
                self.coords_orig = np.asarray(pointcloud.points)
                self.coords = ME.utils.batched_coordinates([self.coords_orig / self.voxel_size], dtype=torch.float32)
                pcd = pointcloud.voxel_down_sample_and_trace(0.1, pointcloud.get_min_bound(), pointcloud.get_max_bound(),
                                                          approximate_class=True)
                self.features=np.ones((self.coords.shape[0],1))
                # cloud_with = self.compute_normals(pointcloud)
                # final_normals = np.zeros((self.coords_orig.shape))
                # for k, t in enumerate(pcd[2]):
                #     final_normals[np.asarray(t)] = np.asarray(cloud_with.normals)[k]
                # self.features = np.stack((final_normals[:, 2], self.coords[:, 2]), axis=1)
                # z_norm = self.normalize_features(self.features[:, 1])
                # self.features[:, 1] = z_norm

                #Inferencia
                test_in_field = ME.TensorField(torch.from_numpy(self.features).to(dtype=torch.float32),
                                               coordinates=self.coords,
                                               quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                               minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                               device=self.device)
                test_output = self.model(test_in_field.sparse())
                logit = test_output.slice(test_in_field)
                pred_raw = logit.F.detach().cpu().numpy()
                pred = np.where(pred_raw > self.threshold, 1, 0)
                stop = time.time()

                points = self.visualize_each_cloud(pred,self.coords_orig)
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "base_link"
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1),
                          PointField('rgb', 16, PointField.UINT32, 1),
                          ]
                self.pc2_trav=pc2.create_cloud(header,fields,points)
                self.pub.publish(self.pc2_trav)
                duration = stop - start
                print(duration)

        self.counter+=1


    def visualize_each_cloud(self,pred, coords):
        points = []
        # SOLUCION CATEGORICA DEL PROBLEMA
        for k,i in enumerate(pred):
            if i == 1:
                r=95
                g=158
                b=160
                a=255
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [coords[k, 0], coords[k, 1], coords[k, 2], rgb]

            if i == 0:
                r=106
                g=90
                b=205
                a=255
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [coords[k, 0], coords[k, 1], coords[k, 2], rgb]

            points.append(pt)
        return points

    def compute_normals(self,pcd):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
        pcd.orient_normals_to_align_with_direction()
        normals = np.asarray(pcd.normals)
        ey = o3d.geometry.PointCloud()
        ey.points = o3d.utility.Vector3dVector(pcd.points)
        ey.normals = o3d.utility.Vector3dVector(normals)
        return ey


    def normalize_features(self,features):
        norm_arr = np.empty_like(features)
        # for dim in range(features.shape[1]):
        minimo = min(features[:])
        diff_arr = max(features[:]) - minimo
        for n, l in enumerate(features[:]):
            norm_arr[n] = ((l - minimo) / diff_arr)
        return norm_arr.astype(np.float32)



if __name__ == '__main__':
    rospy.init_node("traversability_analysis")
    ey = traversability_ROS()
    ey.listener()
    rospy.spin()

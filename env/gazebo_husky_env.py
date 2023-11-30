#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:52:39 2023

@author: Alina Kasiuk
"""

import gym
import os
import sys
import rospy
import roslaunch
import subprocess
import time
import numpy as np
import math

from gym import utils, spaces
from gym.utils import seeding

from envs.gazebo_env import gazebo_env

from geometry_msgs.msg import Point, Twist, PoseStamped, Pose, Vector3Stamped

from std_srvs.srv import Empty

# import lidar point cloud

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class GazeboHuskyEnv(gazebo_env.GazeboEnv):

    def __init__(self): 

        
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/arvc/alina_ws/src/husky_custom_simulation/launch/campus_vfinal.launch")
        
        self.speed = Twist()
        self.pub_vel=rospy.Publisher("/husky_velocity_controller/cmd_vel", Twist, queue_size = 1)
        self.r = rospy.Rate(5)#10Hz
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty) 
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)   
      
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        
        self._seed()
        
        self.gps_vel_x = 0
        self.gps_vel_y = 0
        self.gps_vel_z = 0
        
        self.points = list()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
    
    def listener(self):
            
        self.sub = rospy.Subscriber("/navsat/vel", Vector3Stamped, self.callback_gps_vel)
        self.cloud_sub = rospy.Subscriber("/os1/pointCloud", PointCloud2,self.callback_cloud) 
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")        
        
      
        
    def cb_once(self,msg):
    	#do processing here
    	self.sub.unregister()
    	
    def cb_once_cloud(self,msg):
    	#do processing here
    	self.cloud_sub.unregister()
    	
    def close_listener(self):
        self.sub = rospy.Subscriber("/navsat/vel", Vector3Stamped, self.cb_once)
        self.cloud_sub = rospy.Subscriber("/os1/pointCloud", PointCloud2, self.cb_once_cloud) 	


    def callback_cloud(self, ros_point_cloud):
        print("CALLBACK: PointCloud")
        field_names = [field.name for field in ros_point_cloud.fields]
        self.points = list(pc2.read_points(ros_point_cloud, skip_nans=True, field_names=field_names))

    def callback_gps_vel(self, msg):
        print("CALLBACK: gps_vel")
        self.gps_vel_x = msg.vector.x
        self.gps_vel_y = msg.vector.y
        self.gps_vel_z = msg.vector.z
        
    def get_vel(self):
        real_vel = math.sqrt(self.gps_vel_x**2+self.gps_vel_y**2+self.gps_vel_z**2)
        return real_vel
        
    def get_cloud(self):
        if len(self.points) == 0:
            print("Converting an empty cloud")
            return np.empty(0)
        pcd_array = np.asarray(self.points)
        cloud = pcd_array[:, 0:3]
        return cloud  
   
        
    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        
        if action == 0: #FORWARD
            self.speed.linear.x = 0.3
            self.speed.angular.z = 0.0
            self.pub_vel.publish(self.speed)
            self.r.sleep()
        elif action == 1: #LEFT
            self.speed.linear.x = 0.05
            self.speed.angular.z = 0.5
            self.pub_vel.publish(self.speed)
            self.r.sleep()
        elif action == 2: #RIGHT
            self.speed.linear.x = 0.05
            self.speed.angular.z = -0.5
            self.pub_vel.publish(self.speed)
            self.r.sleep()
        
# TODO: get data from LiDAR
   
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")     
            
        
# TODO: write the cost/reward function as real_vel/cmd_vel
# TODO: done from discretize_observation, data from LiDAR left

# TODO: check the movement with simple commanding
#	 return state, reward, done, {}


    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

            
# TODO: get data from LiDAR
            
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
            
# TODO: get data from LiDAR
            
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


    def close(self):
        gazebo_env.GazeboEnv.close(self)
      	
      	
        

#!/usr/bin/env python
import gym
import envs
from envs.gazebo_husky_env import GazeboHuskyEnv
import time
import numpy as np
import open3d as o3d
import random
import time
from sklearn.preprocessing import normalize
import sys

#import liveplot

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':
    env = GazeboHuskyEnv()
    outdir = '/tmp/gazebo_gym_experiments'

    actions=range(env.action_space.n)

    total_episodes = 2
    
    # TODO: Fix: don't start doing actions until gazebo is loaded

    # TODO: Fix the error
#    env.listener()
    
    for x in range(total_episodes):

        env.reset()
        
        env.listener()
        
        input("Press Enter to continue...")
        print("Episode ", x, " of ", total_episodes)

        for i in range(200):

            # Pick a random action 
            i = random.choice(range(len(actions)))

            action = actions[i] 
            env.step(action)
            real_vel = env.get_vel()
            
            points = env.get_cloud()
#            np.set_printoptions(suppress=True)
            
            if points.size == 0:
            	points = 0
            else:

#           	point_dir = "Points1/Point" + str(x) + "_" + str(i) + ".txt"
#            	points_to_txt = np.savetxt(point_dir, points, fmt="%.3f")

            	print("action ", actions[i] ," done")
            	print("velocity ", real_vel)
            	print("point shape", points.shape)
            	print("___________________")
            	


        # Initialize a visualizer object
       # vis = o3d.visualization.Visualizer()
        # Create a window, name it and scale it
       # vis.create_window(window_name='Bunny Visualize', width=800, height=600)
        # Add the voxel grid to the visualize
       # vis.add_geometry(voxel_grid)
        # We run the visualizater
      #  vis.run()
        # Once the visualizer is closed destroy the window and clean up
      #  vis.destroy_window()
            

#        if x%100==0:
#            plotter.plot(env)

    env.close()

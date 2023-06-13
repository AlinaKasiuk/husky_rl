#!/usr/bin/env python
import gym
import envs
from envs.gazebo_husky_env import GazeboHuskyEnv
import time
import numpy
import random
import time

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
    input("Press Enter to continue...")
    # TODO: Fix the error
    env.listener()
    for x in range(total_episodes):

        env.reset()
        
        print("Episode ", x, " of ", total_episodes)

        for i in range(200):

            # Pick a random action 
            i = random.choice(range(len(actions)))

            action = actions[i] 
            env.step(action)
            real_vel = env.get_vel()
            
            print("action ", actions[i] ," done")
            print("velocity ", real_vel)
            

#        if x%100==0:
#            plotter.plot(env)

    env.close()

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
    print("point 1")
    env = GazeboHuskyEnv()
    print("point 2")
    outdir = '/tmp/gazebo_gym_experiments'
    #env = monitor.Monitor(GazeboHuskyEnv(), outdir, force=True)
#    plotter = liveplot.LivePlot(outdir)
    actions=range(env.action_space.n)
    print("point 2")

    total_episodes = 2
    highest_reward = 0

    for x in range(total_episodes):

        env.reset()
        print("point 3")

        for i in range(200):

            # Pick a random action 
            i = random.choice(range(len(actions)))
            print(actions[i])
            action = actions[i] 
            env.step(action)
            print("action done")

#        if x%100==0:
#            plotter.plot(env)

    env.close()

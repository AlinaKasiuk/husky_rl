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
#TODO:Check all imports
from agents.agent import DeepQAgent

if __name__ == '__main__':


 #TODO: define training params
 
    # training
    batch_size = train_vars['batch_size']
    net_replace_iter = train_vars['net_update']
    # create the agent TODO
    agent = DeepQAgent(img_size, img_size, in_channels, \
        num_classes=len(ACTIONS().actions()),
        model_name=train_vars['model'],
        device=device,
        weight_path=train_vars['weights']
    )
    # replay memory
    replay_memory = []
    total_iters = 0
    save_every = 5000
 
    env = GazeboHuskyEnv()
    
    outdir = '/tmp/gazebo_gym_experiments'

    actions=range(env.action_space.n)

    total_episodes = 2
    
    input("Press Enter to continue...")    
    env.pause()
            
    for x in range(total_episodes):
        
        done = False
        cumulated_reward = 0 #Should going forward give more reward then L/R ?
        observation = env.reset()
           
        #TODO: Join this   
        env.listener()
        env.trav_listener()


        print("Episode ", x, " of ", total_episodes)

        for iter in range(env_iter):

            # Pick a random action 
            i = random.choice(range(len(actions)))
            
            #TODO: add select_action func
            action = select_action(agent, observation, action_eps, actions[i])
            
            env.step(action)
            print("action ", actions[i] ," done")
            real_vel = env.get_vel()

            
            #TODO: add reward function and create this output
            next_observation, reward, done, _ = env_.step(action)
            cumulated_reward += reward
            
             # resize the observation to the image size, and store it in the replay memory
            replay_memory.append((observation, next_observation, action, r, done)) 
             # update the observation value
            observation = next_observation
            
            if epoch % train_every == 0: # time to train
                if len(replay_memory) > max_rep_mem:
                        replay_memory = replay_memory[-max_rep_mem:]
                    # select the training batch
                    rand_indx = np.random.randint(len(replay_memory), size=batch_size)
                    batch = np.array(replay_memory, dtype='object')[rand_indx]

                    agent.train(batch)
                    if agent.train_iterations % net_replace_iter == 0:
                        print("Replacing target net...")
                        agent.replace_target_network()
                if done:
                    break
                	
        env.close_listener()
        env.close_trav_listener()

    env.close()

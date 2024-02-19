#!/usr/bin/env python
import os
import argparse
import numpy as np

from utils.constants import ROOT
from utils.general import select_action, get_str_device
from utils.logger import BasicLogger

from envs.gazebo_husky_env import GazeboHuskyEnv
from envs.env_utils import ACTIONS #TODO
from agents.agent import DeepQAgent

import random #TODO: remove



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1, \
        help='number of time to restart the environment and com back to the initial point')
    parser.add_argument('--env-iter', type=int, default=5, \
        help='total number of movements in a fixed environment')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--train-steps', type=int, default=50, help='train the agent every X steps')
    parser.add_argument('--rep-memory', '--rm', type=int, default=10000, help='replay memory maximum size')
    parser.add_argument('--net-update', type=int, default=500, help='replay memory maximum size')
    
    parser.add_argument('--a-eps', type=float, default=0.6, help='exploration epsilon') #TODO
    parser.add_argument('--weights', type=str, default='None', \
        help='initial weights path. Default is None. If a valid path is provided, model parameter is not used.') 
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=os.path.join(ROOT, 'runs', 'train'), help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    return parser.parse_args()

def main(train_vars):
    total_episodes = train_vars['episodes']
    env_iter = train_vars['env_iter']
 
    max_rep_mem = train_vars['rep_memory']
    action_eps = train_vars['a_eps'] # TODO: Apply epsilon discount
    train_every = train_vars['train_steps']
    device = get_str_device(train_vars['device'])
    
    logger = BasicLogger(train_vars['project'], train_vars['name'])
 
    # training
    batch_size = train_vars['batch_size']
    net_replace_iter = train_vars['net_update']
    agent = DeepQAgent(num_classes=len(ACTIONS().actions()), \
        model_name=train_vars['model'],
        device=device,
        weight_path=train_vars['weights']
    )
        
    # replay memory
    replay_memory = []
    save_every = 100
 
    env = GazeboHuskyEnv()

    input("Press Enter to continue...")    
    env.pause()
            
    for episode in range(total_episodes):
        
        done = False
        cumulated_reward = 0 
        observation = env.reset() #TODO: Return the pointcloud as an observation
        # Coordinates -?
           
        #TODO: Join this   
        env.listener()
        env.trav_listener()


        print("Episode ", episode, " of ", total_episodes)

        for iteration in range(env_iter):

            # Pick an action based on the current state
            action = env.action_space.sample()
            action = select_action(agent, observation, action_eps,  action)
            
            #TODO: add reward function and create this output
            next_observation, reward, done, _ = env.step(action)
            cumulated_reward += reward
            print("action ", action ," done")
            
            replay_memory.append((observation, next_observation, action, reward, done)) 
             # update the observation value
            observation = next_observation
            
            if iteration % train_every == 0: # time to train
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
        if (episode+1) % save_every == 0:
            logger.save_model(agent.model, agent.model_name+str(episode))
                	
        env.close_listener()
        env.close_trav_listener()

    env.close()
    
if __name__ == "__main__":
    opt_ = parse_opt()
    main(vars(opt_))    
    
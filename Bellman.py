#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:51:42 2019

@author: maachou
"""

import gym
import numpy as np
import matplotlib.pyplot as plt 

class bellman:
    def __init__(self,n_states,n_actions,game,gamma,learning_rate,nb_iterations):
        self.n_states=n_states
        self.n_actions=n_actions
        self.Q_table=None
        self.game=game
        self.gamma=gamma
        self.lr=learning_rate
        self.iter=nb_iterations
        
        
        
    def initialize_Q(self):
        self.Q_table=np.zeros((self.n_states,self.n_actions))



    def act_greedy(self,state):
        chosen=None
        best_q=-10e20
        for action in range(6):
            q=self.Q_table[state][action]
            if q>best_q:
                chosen=action
                best_q=q    
        return chosen

    def maximum_expected(self,state):
        best_q=-10e20
        for action in range(6):
            q=self.Q_table[state][action]
            if q>best_q:
                best_q=q 
        return best_q


              
    def train(self):
        self.initialize_Q()
        for i in range(self.iter):
            print(i)
            epsilon=1
            decaying_rate=10
            k=0
            env=gym.make(self.game).env
            done=False
            state=env.s
            while done==False:
                exploration=np.random.rand()
                if exploration<epsilon:
                    action=env.action_space.sample()
                else:
                    action=self.act_greedy(state)
                new_state,reward,done,info=env.step(action)            
                if new_state==state:
                        reward-=10
                if done==True:
                        reward+=60 
                
                self.Q_table[state][action]=self.Q_table[state][action]+self.lr*(reward+self.gamma*self.maximum_expected(new_state)-self.Q_table[state][action])
                k+=1
                state=new_state
                epsilon=epsilon*np.exp((-k/decaying_rate))
                
                
                
    def visualize(self):
        env=gym.make(self.game).env
        state=env.s
        epochs = 0
        penalties, reward = 0, 0
        
        frames = []
        
        done = False

        while not done and epochs<100:
            exploration=np.random.rand()
            if exploration>0.8:
                action=env.action_space.sample()
            else:
                action = self.act_greedy(state)
            state, reward, done, info = env.step(action)
            # Put each rendered frame into the dictionary for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )
        
            epochs += 1
        
        print("Timesteps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))
        
        # Printing all the possible actions, states, rewards.
        from time import sleep
        def framess(frames):
            for i, frame in enumerate(frames):
                
                print(frame['frame'])
                print(f"Timestep: {i + 1}")
                print(f"State: {frame['state']}")
                print(f"Action: {frame['action']}")
                print(f"Reward: {frame['reward']}")
                sleep(.1)
                
        framess(frames)
            
            
                
   
        
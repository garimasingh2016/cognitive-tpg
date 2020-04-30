from __future__ import division
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import json
import datetime as dt
from gonogo import GoNoGo
import pandas as pd
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random

# import to do training
from tpg.trainer import Trainer
# import to run an agent (always needed)
from tpg.agent import Agent

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: GoNoGo()])

import time # for tracking time

tStart = time.time()

# first create an instance of the TpgTrainer
# this creates the whole population and everything
# teamPopSize should realistically be at-least 100
trainer = Trainer(actions=range(2), teamPopSize=20, rTeamPopSize=20) 

curScores = [] # hold scores in a generation
summaryScores = [] # record score summaries for each gen (min, max, avg)

# 5 generations isn't much (not even close), but some improvements
# should be seen.
for gen in range(5): # generation loop
    curScores = [] # new list per gen
    
    agents = trainer.getAgents()
    
    while True: # loop to go through agents
        teamNum = len(agents)
        agent = agents.pop()
        if agent is None:
            break # no more agents, so proceed to next gen
        
        state = env.reset() # get initial state and prep environment
        score = 0
        for i in range(500): # run episodes that last 500 frames
            show_state(env, i, 'Assault', 'Gen #' + str(gen) + 
                       ', Team #' + str(teamNum) +
                       ', Score: ' + str(score)) # render env
            
            # get action from agent
            # must transform to at-least int-32 (for my getState to bitshift correctly)
            act = agent.act(getState(np.array(state, dtype=np.int32))) 

            # feedback from env
            state, reward, isDone, debug = env.step(act)
            score += reward # accumulate reward in score
            if isDone:
                break # end early if losing state

        agent.reward(score) # must reward agent (if didn't already score)
            
        curScores.append(score) # store score
        
        if len(agents) == 0:
            break
            
    # at end of generation, make summary of scores
    summaryScores.append((min(curScores), max(curScores),
                    sum(curScores)/len(curScores))) # min, max, avg
    trainer.evolve()
    
#clear_output(wait=True)
print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
print('Results:\nMin, Max, Avg')
for result in summaryScores:
    print(result[0],result[1],result[2])
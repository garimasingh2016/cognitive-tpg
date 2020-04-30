import gym
from gym import spaces
import numpy as np
import random

MAX_STEPS = 20000

class GoNoGo(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(GoNoGo, self).__init__()
    # Define action and observation space

    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.screen_history = [0, 0]
    self.action_space = spaces.Discrete(2)
    # Current color
    # Previous color
    self.observation_space = spaces.Box(
      low=0, high=1, shape=(1, 2), dtype=np.float16)

  def _take_action(self, action):
    current_screen = random.random()
    self.screen_history.append(current_screen)

    action_type = action
    if action_type < 1:
      # go
      if current_screen < 0.5:
        self.score += 1
      else:
        self.score -= 1
    elif action_type < 2:
      #no go
      if current_screen < 0.5:
        self.score -= 1
      else:
        self.score += 1


  def step(self, action):
    # Execute one time step within the environment
    self._take_action(action)
    self.current_step += 1

    reward = self.score
    done = self.current_step < MAX_STEPS

    obs = self._next_observation()

    return obs, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    self.score = 0
    # Set the current step
    self.current_step = 1

    return self._next_observation()

  def _next_observation(self):
    return np.array([self.screen_history[self.current_step - 1], self.screen_history[self.current_step]])

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print("step: " + str(self.current_step))
    print("score: " + str(self.score))





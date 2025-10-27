# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent

from stable_baselines3 import DQN
#from sb3_contrib import RecurrentPPO # Importing an LSTM
from stable_baselines3.common.monitor import Monitor

from gymnasium.spaces import Discrete, Box
from gymnasium import ActionWrapper
from gymnasium.wrappers import TransformObservation

import numpy as np

class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)

    def _initialize(self) -> None:
        self.new_observation_space = Box(self.observation_space.low[:48], self.observation_space.high[:48])

        if self.file_path is None:

            self.model = CustomDQN(
                "MlpPolicy", 
                CustomActionWrapper(TransformObservation(self.env, transform_obs, self.new_observation_space)), 
                verbose=0
            )
            del self.env
        else:
            self.model = CustomDQN.load(self.file_path)

    def _gdown(self) -> str:
        # data_path = "rl-model.zip"
        # if not os.path.isfile(data_path):
        #     print(f"Downloading {data_path}...")
        #     # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
        #     url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
        #     gdown.download(url, output=data_path, fuzzy=True)
        # return data_path
        return

    def predict(self, obs):
        # convert to new observation space
        obs = transform_obs(obs)

        action, _ = self.model.predict(obs)

        # convert to original action space
        action = CustomActionWrapper.discrete_action_to_keys(action, obs)

        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4, verbose=0):
        self.model.set_env(
            CustomActionWrapper(TransformObservation(env, transform_obs, self.new_observation_space)))
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class CustomDQN(DQN):
    pass

class CustomActionWrapper(ActionWrapper):

    @staticmethod
    def discrete_action_to_keys(action, obs):
        '''Converts our discrete action space to the original multi-binary box action space.
        ORIGINAL ACTION SPACE (Box): w, a, s, d, space, h, l, j, k, g'''

        move_data = action % 6
        att_jmp_dodge_data = action // 6

        # process move_data
        x_move = move_data % 3 # 0=left, 1=nothing, 2=right ('a' & 'd' keys)
        down = move_data // 3 # 0=nothing, 1=pressed ('s' key)

        # process att_jmp_dodge_data
        dodge = 0
        if att_jmp_dodge_data == 6: # dodge
            dodge = 1
            att_jmp_dodge_data = 0

        attack = att_jmp_dodge_data % 3 # 0=nothing, 1=light, 2=heavy ('j' & 'k' keys)
        jump = att_jmp_dodge_data // 3 # 0=nothing, 1=jump ('space' key)

        # spam pickup if no weapon; if weapon, don't press to avoid dropping weapon
        pickup = obs[15] == 0

        return np.array((
            0, # w (aim up)
            x_move == 0, # a (move left)
            down, # s (move down)
            x_move == 2, # d (move right)
            jump, # space (jump)
            pickup, # h (pickup)
            dodge, # l (dodge)
            attack == 1, # j (light attack)
            attack == 2, # k (heavy attack)
            0, # g (taunt)
        ))

    def __init__(self, env):
        super().__init__(env)

        # movement: {left, nothing, right}×{down, nothing} = 3 * 2 = 6 combinations
        # attack/jump/dodge: ({light, heavy, nothing}×{jump, nothing}) ∪ {dodge} = 3*2 + 1 = 7 combinations
        # 6 * 7 = 42 total action combinations
        self.action_space = Discrete(42) # [0, 42]

    def action(self, action):
        return self.discrete_action_to_keys(action, self.observation)

    # store most recent observation in self.observation
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.observation = obs
        return obs, info

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = super().step(*args, **kwargs)
        self.observation = obs
        return obs, reward, terminated, truncated, info

def transform_obs(x):
    return x[:48]

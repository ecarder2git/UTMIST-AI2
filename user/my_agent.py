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

#from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv

from gymnasium.spaces import Discrete, Box
from gymnasium import ActionWrapper, ObservationWrapper

import numpy as np

class SubmittedAgent(Agent):
    N_ENVS = 4

    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)

        self.prev_move_types = [0,0]
        self.move_frames = [0,0]

    def wrap_env(self, env):
        return CustomActionWrapper(
                CustomObservationWrapper(env, self.new_observation_space))
        # return SubprocVecEnv([ lambda: CustomActionWrapper(
        #         CustomObservationWrapper(env, self.new_observation_space))
        #     for i in range(self.N_ENVS)])

    def _initialize(self) -> None:
        self.new_observation_space = CustomObservationWrapper.generate_observation_space(
            self.observation_space.low, 
            self.observation_space.high
        )

        if self.file_path is None:

            self.model = CustomDQN(
                "MlpPolicy", 
                self.wrap_env(self.env), 
                verbose=0,
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
        CustomObservationWrapper._step(self, obs)
        obs = CustomObservationWrapper.observation(self, obs)

        #print(obs[48:])

        action, _ = self.model.predict(obs)

        # convert to original action space
        action = CustomActionWrapper.discrete_action_to_keys(action, obs)

        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4, verbose=0):
        self.model.set_env(self.wrap_env(env))
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class CustomDQN(QRDQN):

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        model_action, state = super().predict(observation, state, episode_start, deterministic)

        # call self._generate_override_action(), override model action if it returns an action
        # code to handle vectorized observations copied from stable baselines source code
        if self.policy.is_vectorized_observation(observation):
            if isinstance(observation, dict):
                n_batch = observation[next(iter(observation.keys()))].shape[0]
            else:
                n_batch = observation.shape[0]

            override_actions = [self._gen_override_action(observation[i]) for i in range(n_batch)]
            action = np.array([ model_action[i] if action is None else action 
                for i, action in enumerate(override_actions)  ])
        else:
            override_action = self._gen_override_action(observation)
            action = model_action if override_action is None else np.array(override_action) 

        return action, state

    def _gen_override_action(self, observation):
        '''Return an action to override the action generated by the model. 
            Return None to use the model action.'''

        pass

class CustomActionWrapper(ActionWrapper):

    @staticmethod
    def discrete_action_to_keys(action, obs):
        '''Converts our discrete action space to the original multi-binary box action space.
        ORIGINAL ACTION SPACE (Box): w, a, s, d, space, h, l, j, k, g'''

        move_data = action % 6
        att_jmp_dodge_data = action // 6
            # attack/jump/dodge: {light, heavy, light aim up, heavy aim up, nothing, jump, dodge} # 7 combinations

        # process move_data
        x_move = move_data % 3 # 0=left, 1=nothing, 2=right ('a' & 'd' keys)
        down = move_data // 3 # 0=nothing, 1=pressed ('s' key)

        # spam pickup if no weapon; if weapon, don't press to avoid dropping weapon
        pickup = obs[15] == 0

        return np.array((
            att_jmp_dodge_data == 2 or att_jmp_dodge_data == 3, # w (aim up)
            x_move == 0, # a (move left)
            down, # s (move down)
            x_move == 2, # d (move right)
            att_jmp_dodge_data == 5, # space (jump)
            pickup, # h (pickup)
            att_jmp_dodge_data == 6, # l (dodge)
            att_jmp_dodge_data == 0 or att_jmp_dodge_data == 2, # j (light attack)
            att_jmp_dodge_data == 1 or att_jmp_dodge_data == 3, # k (heavy attack)
            0, # g (taunt)
        ))

    def __init__(self, env):
        super().__init__(env)

        # movement: {left, nothing, right}×{down, nothing} = 3 * 2 = 6 combinations
        # attack/jump/dodge: {light, heavy, light aim up, heavy aim up, nothing, jump, dodge} # 7 combinations
        # 6 * 7 = 42 total action combinations
        self.action_space = Discrete(42) # [0, 41]

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

class CustomObservationWrapper(ObservationWrapper):
    CROP_INDEX = 48

    MAX_MOVE_FRAMES = 3 * 30
    grounded_dodge_cooldown = 30
    air_dodge_cooldown = 82
    MAX_DODGE_COOLDOWN = max(grounded_dodge_cooldown, air_dodge_cooldown)

    obs_additional_low = np.array([0,0,0,0], dtype=np.float32)
    obs_additional_high = np.array([MAX_MOVE_FRAMES, MAX_MOVE_FRAMES, MAX_DODGE_COOLDOWN, MAX_DODGE_COOLDOWN], dtype=np.float32)

    @staticmethod 
    def generate_observation_space(low, high):
        return Box(
            np.concatenate((low[:CustomObservationWrapper.CROP_INDEX], CustomObservationWrapper.obs_additional_low)), 
            np.concatenate((high[:CustomObservationWrapper.CROP_INDEX], CustomObservationWrapper.obs_additional_high))
        )

    def __init__(self, env, observation_space):
        super().__init__(env)
        self.observation_space = observation_space 

        self.prev_move_types = [0,0]
        self.move_frames = [0,0]

        self.dodge_cooldowns = [0, 0]

    def observation(self, obs):
        return np.concatenate((obs[:CustomObservationWrapper.CROP_INDEX], 
            np.array(self.move_frames), np.array(self.dodge_cooldowns)))

    # store most recent observation in self.observation
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self.prev_move_types = [0,0]
        self.move_frames = [0,0]

        self.dodge_cooldowns = [0, 0]

        return obs, info

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = super().step(*args, **kwargs)
        self._step(obs)
        return obs, reward, terminated, truncated, info
    
    def _step(self, obs):
        move_type_obs_is = (14, 46)
        dodge_timer_obs_is = (10, 42)
        grounded_obs_is = (5, 37)

        for i in range(2):
            move_type = obs[move_type_obs_is[i]]

            if move_type != 0 and move_type == self.prev_move_types[i]:
                self.move_frames[i] += 1
                self.move_frames[i] = min(self.move_frames[i], CustomObservationWrapper.MAX_MOVE_FRAMES)
            else: 
                self.move_frames[i] = 0
                self.prev_move_types[i] = move_type

            if self.dodge_cooldowns[i] != 0:
                self.dodge_cooldowns[i] -= 1
            elif obs[dodge_timer_obs_is[i]] != 0:
                if obs[grounded_obs_is[i]] == 1:
                    self.dodge_cooldowns[i] = CustomObservationWrapper.grounded_dodge_cooldown
                else:
                    self.dodge_cooldowns[i] = CustomObservationWrapper.air_dodge_cooldown
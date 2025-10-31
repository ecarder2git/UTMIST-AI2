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
from environment.agent import Agent, CustomActionWrapper, CustomObservationWrapper

#from stable_baselines3 import DQN
from sb3_contrib import QRDQN

import numpy as np

from functools import partial # vec

class SubmittedAgent(Agent):
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

    #def wrap_env(self, env):   # Vec: wrap_env can't be done inside submittedAgent. It is done in agent.py.train()
    #    return CustomActionWrapper(
    #            CustomObservationWrapper(env, self.new_observation_space))
    #    # return SubprocVecEnv([ lambda: CustomActionWrapper(
    #    #         TransformObservation(env, transform_obs, self.new_observation_space))
    #    #     for i in range(self.N_ENVS)])

    def _initialize(self) -> None:
        #self.new_observation_space = CustomObservationWrapper.generate_observation_space( # Vec: This is done in train_agent
        #    self.observation_space.low, 
        #    self.observation_space.high
        #)

        print("INITALIZATION",self.file_path) # vec: useful to know when a new agent is created
        if self.file_path is None:

            self.model = CustomDQN(
                "MlpPolicy", 
                self.env,#self.wrap_env(self.env), # Vec: wrap_env is moved
                verbose=0,
                exploration_initial_eps=1.0,     # Vec: I don't like default epxloration rates
                exploration_fraction=0.1,
                exploration_final_eps=0.03,
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

    def predict(self, obs, getRaw=False):
        # convert to new observation space
        CustomObservationWrapper._step(self, obs)                  # vec: this is fine
        obs = CustomObservationWrapper.observation(self, obs)
        #obs = transform_obs(obs)

        #print(obs[48:])

        action, _ = self.model.predict(obs)

        # convert to original action space
        if not getRaw:  #vec: occasionally need the non-transformed version
            action = CustomActionWrapper.discrete_action_to_keys(action, obs)

        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4, verbose=0):
        self.model.set_env(env)#self.wrap_env(env)) # vec : obv
        self.model.verbose = verbose

        self.vecBeforeLearning(env) # vec : need to setup stuff for saving

        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval, reset_num_timesteps=False)

    def vecBeforeLearning(self, env):
        # updating env with oppositions
        self.model.actual_opps = [None] * self.model.n_envs
        self.model.opp_agents = [] # these shouldn't be recursive, since learn(...) is only called on the original
        for partPath in []:#["v4999","v9999","v14999","v19999"]:
            #add to self.opp_agents
            #if path:
            path = "checkpoints/alpha_1/" + partPath + ".zip"
            print(f"Updating Opponent List with previously trained model(s) at: {path}")
            try:
                opponent = (partial(SubmittedAgent))(file_path=path)
            except FileNotFoundError:
                print(f"Warning: Self-play file {path} not found. ADJUST LIST IN OFF_POLICY_ALGORITHM.py OR ELSE.")
            #else:
            #    print("Warning: No self-play model saved. Defaulting to constant agent.")
            #    opponent = ConstantAgent()
            opponent.get_env_info(env)
            self.model.opp_agents.append(opponent)

        #print("MARKING")
        #print(self.model.exploration_rate)
        #print("Marking22")
        #for i in self.model.opp_agents:
        #    try:
        #        print(i.model.exploration_rate, i)
        #    except:
        #        print("No exploration rate", i)

        #print(self.model.n_envs)
        for i in range(self.model.n_envs):
            if np.random.random() < 0.3: # - Change in Offpolicyalgorithm.py for actual change. This one is irrelevant
                self.model.actual_opps[i] = None # None represents constant_agent
            else:
                try:
                    self.model.actual_opps[i] = np.random.choice(self.model.opp_agents)
                except:
                    self.model.actual_opps[i] = None
        #print(self.model.actual_opps)
        #exit()

        self.model.subAgent = self

    def updateOpponentList(self, filepath):
        print(f"Updating Opponent List with newly saved model at: {filepath}")
        path = filepath
        try:
            opponent = (partial(SubmittedAgent))(file_path=path)
        except FileNotFoundError:
            print(f"Warning: Self-play file {path} not found. ADJUST LIST IN OFF_POLICY_ALGORITHM.py OR ELSE.")
        #else:
        #    print("Warning: No self-play model saved. Defaulting to constant agent.")
        #    opponent = ConstantAgent()
        opponent.get_env_info(self.model.env)
        self.model.opp_agents.append(opponent)
        
        #print("MARKING")
        #print(self.model.exploration_rate)
        #print("Marking22")
        #for i in self.model.opp_agents:
        #    try:
        #        print(i.model.exploration_rate, i)
        #    except:
        #        print("No exploration rate", i)

        #for i in self.model.opp_agents:
        #    print(i.model.predict(np.zeros(48)))
        #exit()

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

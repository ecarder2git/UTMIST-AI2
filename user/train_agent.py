'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

from user.my_agent import *

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def danger_zone_sides_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_width: float = 7.1
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if abs(player.body.position.x) >= zone_width else 0.0

    return reward * env.dt

def danger_zone_high_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = -3.0
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y <= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty/reward for being in a specific state.

    Args:
        env (WarehouseBrawl): The game environment.
        desired_state (Type[PlayerObjectState]): The state to check against.

    Returns:
        float: The computed penalty/reward as a tensor.
    """
    
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,    
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain width threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_width (float): The width threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """

    # multiple = 1 if self.body.position.x < 0 else -1
        # self.env.add_reward(self.agent_id, multiple * (self.body.position.x - self.prev_x))
        
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply reward/penalty based on movement towards/away from opponent
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

# ---------------------------------------------------------------------------------
# ----------------------------- CUSTOM REWARD FUNCTIONS ---------------------------
# ---------------------------------------------------------------------------------

def head_to_weapon_reward(
    env: WarehouseBrawl,
) -> float:
    
    """Reward for moving towards the weapon on the ground."""
    class Weapon_Spawner():
        def __init__(self, x, y, weapon_type):
            self.x = x
            self.y = y
            self.weapon_type = weapon_type

        def __lt__(self, other):
            dist_self = (self.x - player.body.position.x)**2 + (self.y - player.body.position.y)**2
            dist_other = (other.x - player.body.position.x)**2 + (other.y - player.body.position.y)**2
            return dist_self < dist_other
        
    player = env.objects["player"]

    if player.weapon != "Punch":
        return 0.0  # No reward if already holding a weapon
    
    # Finding the nearest weapon spawner with an active weapon
    obs = player.get_obs()
    weapon_1 = Weapon_Spawner(obs[16], obs[17], obs[18])
    weapon_2 = Weapon_Spawner(obs[19], obs[20], obs[21])
    weapon_3 = Weapon_Spawner(obs[22], obs[23], obs[24])
    weapon_4 = Weapon_Spawner(obs[25], obs[26], obs[27])

    weapons = [weapon_1, weapon_2, weapon_3, weapon_4]
    active_weapons = [w for w in weapons if w.weapon_type != 0]

    nearest_weapon = min(active_weapons) if active_weapons else None

    if nearest_weapon is None:
        return 0.0  # No active weapons available
    
    # Calculate reward based on movement towards the nearest weapon
    prev_distance = ((player.prev_x - nearest_weapon.x) ** 2 + (player.prev_y - nearest_weapon.y) ** 2) ** 0.5
    current_distance = ((player.body.position.x - nearest_weapon.x) ** 2 + (player.body.position.y - nearest_weapon.y) ** 2) ** 0.5

    reward = prev_distance - current_distance
    return reward
            

def dodge_reward(
    env: WarehouseBrawl,
    proximity_x: float = 5.0,
    proximity_y: float = 3.0,
    DODGE_FRAMES: int = 10,
) -> float:
    
    """Reward for successfully dodging an opponent's attack.
    
        Modified from: https://github.com/StellarLuminosity/AI2/blob/main/env_final.ipynb
    """

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Ensures the reward is only given when the player is within proximity to the opponent
    # Avoids rewarding dodges that are too far away to be meaningful, which would waste the dodge cooldown
    is_close_proximity_x = abs(player.body.position.x - opponent.body.position.x) < proximity_x
    is_close_proximity_y = abs(player.body.position.y - opponent.body.position.y) < proximity_y

    # Check if the opponent is attacking and the player is in a dodge state
    if isinstance(opponent.state, AttackState) and isinstance(player.state, DodgeState):
        if is_close_proximity_x and is_close_proximity_y and player.damage_taken_this_frame == 0:
            return 1.0 / DODGE_FRAMES  # Reward for dodging
        
    return 0.0
        
def low_health_damage_penalty(
    env: WarehouseBrawl,
    low_health_threshold: float = 50,
) -> float:
    
    """Penalty for taking damage when health is below a certain threshold."""

    player: Player = env.objects["player"]

    # Apply penalty if health is below the threshold and damage is taken
    if player.health >= low_health_threshold:
        return -player.damage_taken_this_frame
    return 0.0

def edge_guard_reward(
    env: WarehouseBrawl,
)   -> float:
    
    "Reward for successfully edge-guarding an opponent."

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = 0

    LEFT_EDGE_X = -7.22
    LEFT_EDGE_Y = 2.45

    RIGHT_EDGE_X = 7.08
    RIGHT_EDGE_Y = 0.45

    if opponent.body.position.x < LEFT_EDGE_X and opponent.body.position.y < LEFT_EDGE_Y:
        if player.body.position.x < opponent.body.position.x + 2.0:
            reward = 1.0 # Reward for edge-guarding on the left side
    elif opponent.body.position.x > RIGHT_EDGE_X and opponent.body.position.y < RIGHT_EDGE_Y:
        if player.body.position.x > opponent.body.position.x - 2.0:
            reward = 1.0 # Reward for edge-guarding on the right side
        
    return reward * env.dt

# ---------------------------------------------------------------------------------
# ----------------------------- SIGNAL REWARDS ------------------------------------
# ---------------------------------------------------------------------------------

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 1.5
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.05, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=30/10),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1/10),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=15/10),
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=10/10),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-1/10, params={'desired_state': AttackState}),
        
        # Custom Rewards
        #'head_to_weapon_reward': RewTerm(func=head_to_weapon_reward, weight=0.03),
        'danger_zone_high_reward': RewTerm(func=danger_zone_high_reward, weight=20/10),
        'danger_zone_sides_reward': RewTerm(func=danger_zone_sides_reward, weight=20/10),
        #'dodge_reward': RewTerm(func=dodge_reward, weight=0.3/10),
        'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=10/10),

        # 'low_health_damage_penalty': RewTerm(func=low_health_damage_penalty, weight=0.5),
        #'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=20/10)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=100/10)),
        #'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5/10)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=20/10)),
        #'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    my_agent = SubmittedAgent(file_path=None)#"checkpoints/alpha_1/v_12.zip")

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    #reward_manager = gen_reward_manager()
    # Self-play settings
    #selfplay_handler = SelfPlayRandom(
    #    partial(type(my_agent)), # Agent class and its keyword arguments
    #                             # type(my_agent) = Agent class
    #)

    # Set save settings here:
    #save_handler = SaveHandler(
    #    agent=my_agent, # Agent to save
    #    save_freq=20_000, # Save frequency
    #    max_saved=40, # Maximum number of saved models
    #    save_path='checkpoints', # Save path
    #    run_name='experiment_1',
    #    mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    #)

    # Set opponent settings here:
    #opponent_specification = {     # Irrelevant
    #                'self_play': (8, selfplay_handler),
    #                'constant_agent': (0.5, partial(ConstantAgent)),
    #                'based_agent': (1.5, partial(BasedAgent)),
    #            }
    #opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        lambda : gen_reward_manager(),
        lambda i : SaveHandler(
                agent=my_agent, # Agent to save
                save_freq=100_000, # Save frequency
                max_saved=40, # Maximum number of saved models
                save_path='checkpoints', # Save path
                run_name='experiment_alpha_' + str(i),
                mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
            ),
        None,
        CameraResolution.LOW,
        train_timesteps=10_000_000,
        train_logging=TrainLogging.PLOT
    )
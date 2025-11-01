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
from environment.environment import RenderMode, CameraResolution
from environment.environment import WarehouseBrawl, Power, Cast, Capsule, CapsuleCollider, MoveType
from environment.agent import run_real_time_match, Agent, CustomObservationWrapper, CustomActionWrapper
from user.train_agent import *#

#from stable_baselines3 import DQN
from sb3_contrib import QRDQN

import numpy as np
import math

from functools import partial # vec

class SubmittedAgent(Agent):
    
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        self.prev_move_types = [0,0]
        self.move_frames = [0,0]

        self.dodge_cooldowns = [0, 0]


        

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
        
        self.model.opp_agents = [] #["v_499",...]: # pretrained opps
        # these shouldn't be recursive, since learn(...) is only called on the original
            
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
            self.model.actual_opps[i] = None
            #if np.random.random() < 0.3: # - Change in Offpolicyalgorithm.py for actual change. This one is irrelevant
            #    self.model.actual_opps[i] = None # None represents constant_agent
            #else:
            #    try:
            #        self.model.actual_opps[i] = np.random.choice(self.model.opp_agents)
            #    except:
            #        self.model.actual_opps[i] = None
        #print(self.model.actual_opps)
        #exit()

        self.model.subAgent = self
        self.model.subAgentClass = SubmittedAgent

    def updateOpponentList(self, filepath):
        print(f"Updating Opponent List with newly saved model at: {filepath}")
        path = filepath
        self.model.opp_agents.append(filepath)


        #try:
        #    opponent = (partial(SubmittedAgent))(file_path=path)
        #except FileNotFoundError:
        #    print(f"Warning: Self-play file {path} not found. ADJUST LIST IN OFF_POLICY_ALGORITHM.py OR ELSE.")
        #else:
        #    print("Warning: No self-play model saved. Defaulting to constant agent.")
        #    opponent = ConstantAgent()
        #opponent.get_env_info(self.model.env)
        #self.model.opp_agents.append(opponent)
    













class CustomDQN(QRDQN):
    dt = 1 / 30.0

    # BRAWL_TO_UNITS = 1.024 / 320
    player_hurtbox_size = 0.928, 1.024 # (290//2, 320//2) * 2 * BRAWL_TO_UNITS
    player_collider_size = 0.87, 1

    ground_heights = 2.85, 0.85
    ground_lefts = -7.0, 2.0
    ground_rights = -2.0, 7.0

    # accel; don't multiply by dt
    x_friction = 0.23888889 # towards 0
    y_friction = 0.5936 # always downwards; positive; includes gravity

    jump_vel = -8.9 # velocity is set to this on jump frame; friction is still applied afterwards

    max_fall_speed = 12
    move_speed = 6.75

    player_bounce_coef = 0.7

    string_max_dodge_window = 8

    skip_steps = 10
    pred_steps = 10

    def __init__(self, policy, env, learning_rate = 0.00005, buffer_size = 1000000, learning_starts = 100, batch_size = 32, tau = 1, gamma = 0.99, train_freq = 4, gradient_steps = 1, replay_buffer_class = None, replay_buffer_kwargs = None, optimize_memory_usage = False, target_update_interval = 10000, exploration_fraction = 0.005, exploration_initial_eps = 1, exploration_final_eps = 0.01, max_grad_norm = None, stats_window_size = 100, tensorboard_log = None, policy_kwargs = None, verbose = 0, seed = None, device = "auto", _init_setup_model = True):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

        
        # attack data; loaded in self._initialize()
        self.keys = None
        self.attacks = None
        self.spear_attacks = None
        self.hammer_attacks = None

        WarehouseBrawl.load_attacks(self)
        self.weapon_attacks_mapping = self.attacks, self.spear_attacks, self.hammer_attacks
    


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
    
    
    def _gen_override_action(self, obs):
        '''Return an action to override the action generated by the model. 
            Return None to use the model action.'''

        dodge_cooldown = int(obs[50])
        stun_frames = int(obs[11])
        move_type = int(obs[14])

        if stun_frames != 0 or move_type != 0: 
            return None # can't do anything

        opp_stun_frames = int(obs[43])
        opp_dodge_timer = int(obs[42])
        opp_move_type = int(obs[46])

        if dodge_cooldown == 0 and opp_move_type != 0:
            dodge_override = self.handle_automatic_dodging(obs)
            if dodge_override is not None: return dodge_override

        if opp_stun_frames != 0 or opp_dodge_timer != 0 or opp_move_type != 0:
            guarenteed_hit = self.find_guarenteed_hit_attack(obs)
            if guarenteed_hit is None: return None

            facing_dir, move_dir_type, is_heavy, jump = guarenteed_hit
            return 6*(5 if jump else 2*(move_dir_type == 0) + is_heavy) + 3*(move_dir_type == 1) + facing_dir+1

        # TODO: opponent also can't do anything if they are in an attack; find guarenteed hits there too
        #   check first each frame to make sure their attack doesn't hit us

        return None


    def find_guarenteed_hit_attack(self, obs):
        pos = obs[0:2]
        vel = obs[2:4]
        jumps_left = int(obs[7])
        weapon_type = int(obs[15])
        grounded = int(obs[5])

        opp_pos = obs[32:34]
        opp_vel = obs[34:36]
        opp_stun_frames = int(obs[43])
        opp_dodge_cooldown = int(obs[51])
        opp_dodge_timer = int(obs[42])
        opp_jumps_left = int(obs[39])
        opp_move_type = int(obs[46])
        opp_weapon_type = int(obs[47])
        opp_move_frame = int(obs[49])

        opp_facing_dir = int(obs[36])
        if opp_facing_dir == 0: opp_facing_dir = -1

        # simulate and cache opponent's future positions up to start of string range
        pred_opp_positions = []

        pred_opp_pos = opp_pos.copy()
        pred_opp_vel = opp_vel.copy()

        frames_progressed = 0

        opp_power = None

        if opp_move_type != 0:
            opp_move = self.weapon_attacks_mapping[opp_weapon_type][MoveType(opp_move_type)]
            opp_power = Power.get_power(opp_move['powers'][opp_move['move']['initialPowerIndex']])

            for i in range(opp_move_frame):
                # if moved is charged, can't know how long they held -> can't make prediction
                if opp_power.is_charge: return None
                opp_power = self.get_next_power(opp_power, opp_move)
                if opp_power is None: return None

        pred_opp_hitboxes = []

        while opp_stun_frames > frames_progressed or opp_dodge_timer > frames_progressed or opp_power is not None:
            pred_opp_positions.append(pred_opp_pos.copy())

            if opp_dodge_timer > frames_progressed: # dodging (staying in place); position is fixed
                pred_opp_hitboxes.append(None)
            else:
                if opp_power is not None:
                    # if moved is charged, can't know how long they held -> can't make prediction
                    if opp_power.is_charge: return None

                    pred_opp_vel = self.update_player_velocity_with_power(opp_power, pred_opp_vel, opp_facing_dir)

                    # add hitboxes
                    current_cast = opp_power.casts[opp_power.cast_idx]
                    in_startup = current_cast.frame_idx < current_cast.startup_frames
                    in_attack = not in_startup and current_cast.frame_idx < (current_cast.startup_frames + current_cast.attack_frames)

                    if not in_attack: pred_opp_hitboxes.append(None)
                    else: pred_opp_hitboxes.append(current_cast.hitboxes)

                    opp_power = self.get_next_power(opp_power, opp_move)
                    if opp_power is None: break
                else: pred_opp_hitboxes.append(None)

                pred_opp_pos, pred_opp_vel = self.player_physics_update(pred_opp_pos, pred_opp_vel, 
                    opp_power is not None, opp_power is not None and opp_power.enable_floor_drag, True) #not power.disable_caster_gravity)
                        # THEY DIDN'T IMPLEMENT disable_caster_gravity CORRECTLY! doesn't actually do anything

            frames_progressed += 1

        pred_opp_positions.append(pred_opp_pos.copy())
        pred_opp_hitboxes.append(None)

        # simulate and cache opponent's future positions in string range, with and without jump
        pred_opp_positions_string = ([], []) # 0 = without jump; 1 = with jump

        for opp_jump in range((opp_jumps_left > 0) + 1): # has to hit even if they jump
            pred_opp_pos_string = pred_opp_pos.copy()
            pred_opp_vel_string = pred_opp_vel.copy()
            frames_progressed_string = frames_progressed

            pred_opp_vel_string[1] += opp_jump * self.jump_vel\

            while opp_dodge_cooldown > frames_progressed_string and \
                frames_progressed_string - opp_stun_frames < self.string_max_dodge_window:

                pred_opp_pos_string, pred_opp_vel_string = self.player_physics_update(pred_opp_pos_string, pred_opp_vel_string)
                pred_opp_positions_string[opp_jump].append(pred_opp_pos_string.copy())

                frames_progressed_string += 1

        string_attack = None
            
        for jump in range((jumps_left > 0) + 1): # test jump-attacking as an option if possible

            for is_heavy in (1, 0): # 0=light, 1=heavy

                for facing_dir in (-1, 1): # test jump-attacking as an option if possible
                    if pos[0] > 0: facing_dir *= -1 # favor direction that knocks opp outwards

                    for move_dir_type in (1,0,2): # 0=neutral, 1=down, 2=side

                        move_grounded = grounded and not jump
                        move_air = not move_grounded
                        move_type = 2 + 6*move_air + 3*is_heavy + move_dir_type
                        if move_type >= 12: move_type -= 1 # since neutral and side recoveries are the same

                        move = self.weapon_attacks_mapping[weapon_type][MoveType(move_type)]

                        power = Power.get_power(move['powers'][move['move']['initialPowerIndex']])

                        pred_pos = pos.copy()
                        pred_vel = vel.copy()

                        frames_progressed = 0

                        # progress game to time of opp stun wearing off and not dodging

                        # do jump if jump; simulate one frame
                        if jump:
                            pred_vel[1] += self.jump_vel
                            pred_pos, pred_vel = self.player_physics_update(pred_pos, pred_vel)
                            frames_progressed += 1

                        # takes a frame after input for attack to actually begin; move forward 1 frame
                        pred_pos, pred_vel = self.player_physics_update(pred_pos, pred_vel)
                        frames_progressed += 1

                        attack_good = True
                        
                        while frames_progressed < len(pred_opp_positions):
                            # charges; assume charging for min amount
                            if power.is_charge and power.total_frame_count > power.min_charge + 1:
                                power = Power.get_power(move['powers'][power.on_miss_next_power_index])
                            
                            else:
                                power.total_frame_count += 1
                                pred_vel = self.update_player_velocity_with_power(power, pred_vel, facing_dir)

                                # check for collision with opp attack hitboxes
                                opp_hitboxes = pred_opp_hitboxes[frames_progressed]

                                if opp_hitboxes is not None and \
                                    self.check_hitboxes_hit(opp_hitboxes, opp_facing_dir, pred_opp_positions[frames_progressed], pred_pos):
                                    attack_good = False
                                    #print(frames_progressed, "bad", facing_dir, ('n','d','s')[move_dir_type], ('L','H')[is_heavy], ('','J')[jump])
                                    break # this attack is no good; runs into opp attack

                                # check for hit at last pred frame = first frame without invulnerability
                                if (opp_hitboxes is not None or frames_progressed == len(pred_opp_positions) - 1) and \
                                    self.check_power_hit(power, facing_dir, pred_pos, pred_opp_positions[frames_progressed]):

                                    #print(frames_progressed, "Combo", facing_dir, ('n','d','s')[move_dir_type], ('L','H')[is_heavy], ('','J')[jump])
                                    return facing_dir, move_dir_type, is_heavy, jump # combo works; favor combo over string
                    
                                power = self.get_next_power(power, move)
                                if power is None: break

                            pred_pos, pred_vel = self.player_physics_update(pred_pos, pred_vel, 
                                True, power.enable_floor_drag, True) #not power.disable_caster_gravity)
                                    # THEY DIDN'T IMPLEMENT disable_caster_gravity CORRECTLY! doesn't actually do anything

                            frames_progressed += 1

                        if not attack_good: continue

                        # check if valid as a string; opp must not have a dodge
                        if power is None: continue
                        if string_attack is not None: continue

                        string_hits = [ False, False ] # without jump, with jump

                        for i in range(len(pred_opp_positions_string[0])): # has to hit even if they jump
                            # charges; assume charging for min amount
                            if power.is_charge and power.total_frame_count > power.min_charge + 1:
                                power = Power.get_power(move['powers'][power.on_miss_next_power_index])
                            
                            else:
                                power.total_frame_count += 1
                                pred_vel = self.update_player_velocity_with_power(power, pred_vel, facing_dir)

                                for jump in range((opp_jumps_left > 0) + 1):
                                    if string_hits[jump]: continue

                                    string_hits[jump] = self.check_power_hit(power, facing_dir, pred_pos, 
                                        pred_opp_positions_string[jump][i])

                                if string_hits[0] and (opp_jumps_left == 0 or string_hits[1]): 
                                    # string will hit in both cases; success
                                    string_attack = facing_dir, move_dir_type, is_heavy, jump
                                    break
                    
                                power = self.get_next_power(power, move)
                                if power is None: break

                            pred_pos, pred_vel = self.player_physics_update(pred_pos, pred_vel, 
                                True, power.enable_floor_drag, True) #not power.disable_caster_gravity)
                                    # THEY DIDN'T IMPLEMENT disable_caster_gravity CORRECTLY! doesn't actually do anything

                            frames_progressed += 1
        
        #if string_attack is not None:
        #    print("String", string_attack[0], ('n','d','s')[string_attack[1]], ('L','H')[string_attack[2]], ('','J')[string_attack[3]])
        return string_attack



    def handle_automatic_dodging(self, obs):
        pos = obs[0:2]
        vel = obs[2:4]
        stun_frames = int(obs[11])
        move_type = int(obs[14])
        jumps_left = int(obs[7])

        if stun_frames != 0 or move_type != 0: 
            return None # can't do anything

        opp_pos = obs[32:34]
        opp_vel = obs[34:36]
        opp_move_type = int(obs[46])
        opp_move_frame = int(obs[49])
        opp_weapon_type = int(obs[47])

        opp_facing_dir = int(obs[36])
        if opp_facing_dir == 0: opp_facing_dir = -1

        move = self.weapon_attacks_mapping[opp_weapon_type][MoveType(opp_move_type)]

        # find power at move_frame by repeatedly updating
        power = Power.get_power(move['powers'][move['move']['initialPowerIndex']])

        for i in range(opp_move_frame):
            # if moved is charged, can't know how long they held -> can't make prediction
            if power.is_charge: return None
            power = self.get_next_power(power, move)
            if power is None: return None

        # simulate one frame forwards
        pred_opp_vel = self.update_player_velocity_with_power(power, opp_vel, opp_facing_dir)

        power = self.get_next_power(power, move)
        if power is None: return None

        pred_opp_pos, _ = self.player_physics_update(opp_pos, pred_opp_vel, 
            True, power.enable_floor_drag, True) #not power.disable_caster_gravity)
                # THEY DIDN'T IMPLEMENT disable_caster_gravity CORRECTLY! doesn't actually do anything

        jump = False
        pred_vel = vel.copy()
        if jumps_left != 0 and jump: pred_vel[1] += self.jump_vel
        pred_pos, _ = self.player_physics_update(pos, pred_vel)

        # check if attack will hit
        if self.check_power_hit(power, opp_facing_dir, pred_opp_pos, pred_pos): 
            return 37 # force dodge if its opponent attack is going to hit

        return None

    # NOTE: not perfect when attack hits the gounrd; 
    #   prediction assumes player will slide along gorund
    #   player actually rams into ground and does not slide
    # unarmed:
    # spear: ssig slight? x, y; dsig slight x, slight more y
    # hammer: dsig???

    def get_next_power(self, power, move):
        current_cast: Cast = power.casts[power.cast_idx]

        in_startup = current_cast.frame_idx < current_cast.startup_frames
        in_attack = not in_startup and current_cast.frame_idx < (current_cast.startup_frames + current_cast.attack_frames)

        if in_attack and  power.cast_idx == len(power.casts) - 1 and power.last_power:
            power.frames_into_recovery += 1

        current_cast.frame_idx += 1

        power.in_recovery = not in_attack and not in_startup
        if not power.in_recovery: return power


        ### in recovery ###

        if power.cast_idx < len(power.casts) - 1: # not last cast; go to next cast
            power.cast_idx += 1
            return power

        # else: handle last cast

        if power.frames_into_recovery < power.recovery_frames: # not done recovering yet; yield until done
            power.frames_into_recovery += 1 
            return power

        # else: done recovering

        if power.last_power: # attack done; end attack
            return None

        # else: still more powers in attack; go to the next power

        # we don't care about further calculations if hit; code would've returned before reachinh here
        # if power.hit_anyone and power.on_hit_next_power_index != -1: # go to on_hit_next_power
        #     return Power.get_power(move['powers'][power.on_hit_next_power_index])

        if power.on_miss_next_power_index != -1: # go to on_miss_next_power
            return Power.get_power(move['powers'][power.on_miss_next_power_index])

        return power # stay on the same power

    def update_player_velocity_with_power(self, power: Power, vel, facing_dir):
        # if power.allow_left_right_mobility:
        #     # assume always holding sideways move key
        #     vel[0] += min(self.x_friction, self.move_speed) * facing_dir 

        current_cast = power.casts[power.cast_idx]
        changes = current_cast.get_frame_data(current_cast.frame_idx)

        if changes is None:
            return vel

        n_vel = vel.copy()

        cvs = changes.caster_velocity_set
        #print(f"{self.agent_id} {cvs} ding!")
        if cvs is not None and cvs.active:
            angle_rad = math.radians(cvs.directionDeg)
            n_vel = np.array((math.cos(angle_rad), -math.sin(angle_rad)))  * cvs.magnitude
            n_vel[0] *= int(facing_dir)
        
        # Process caster velocity damp XY.
        cvdxy = changes.caster_velocity_damp_xy
        if cvdxy is not None:
            if getattr(cvdxy, 'activeX', False): n_vel[0] *= cvdxy.dampX
            if getattr(cvdxy, 'activeY', False): n_vel[1] *= cvdxy.dampY

        # Process caster velocity set XY.
        cvsxy = changes.caster_velocity_set_xy
        if cvsxy is not None:
            if getattr(cvsxy, 'activeX', False): n_vel[0] = cvsxy.magnitudeX * int(facing_dir)
            if getattr(cvsxy, 'activeY', False): n_vel[1] = cvsxy.magnitudeY
        
        # Process caster velocity add XY.
        cvaxy = changes.caster_velocity_add_xy
        if cvaxy is not None:
            if getattr(cvaxy, 'activeX', False): n_vel[0] += cvaxy.magnitudeX * int(facing_dir)
            if getattr(cvaxy, 'activeY', False): n_vel[1] += cvaxy.magnitudeY

        return n_vel

    def check_power_hit(self, power, facing_dir, player_pos, opponent_pos):
        current_cast = power.casts[power.cast_idx]
        in_startup = current_cast.frame_idx < current_cast.startup_frames
        in_attack = not in_startup and current_cast.frame_idx < (current_cast.startup_frames + current_cast.attack_frames)

        if not in_attack:
            return False

        return self.check_hitboxes_hit(current_cast.hitboxes, facing_dir, player_pos, opponent_pos)

    def check_hitboxes_hit(self, hitboxes, facing_dir, player_pos, opponent_pos):
  
        for hitbox in hitboxes:
            hitbox_offset = Capsule.get_hitbox_offset(hitbox['xOffset'], hitbox['yOffset'])
            hitbox_offset = (hitbox_offset[0] * facing_dir, hitbox_offset[1]) # adjust for facing direction

            hitbox_pos = (player_pos[0] + hitbox_offset[0], player_pos[1] + hitbox_offset[1])
            hitbox_size = Capsule.get_hitbox_size(hitbox['width'], hitbox['height'])
            hitbox_collider = CapsuleCollider(center=hitbox_pos, width=hitbox_size[0], height=hitbox_size[1])

            hurtbox_collider = CapsuleCollider(center=tuple(opponent_pos), 
                width=self.player_hurtbox_size[0], height=self.player_hurtbox_size[1])

            intersects = hurtbox_collider.intersects(hitbox_collider)
           
            if intersects:
                return True

        return False

    def player_physics_update(self, pos, vel, attacking=False, apply_x_friction=True, apply_y_friction=True):
        n_vel = vel.copy()

        if apply_x_friction:
            n_vel[0] -= min(abs(n_vel[0]), self.x_friction) * np.sign(n_vel[0])

        n_pos = pos + n_vel*self.dt

        # handle collision with ground
        on_ground = False

        bot = n_pos[1] + self.player_collider_size[1]/2
        left = n_pos[0] - self.player_collider_size[0]/2 
        right = n_pos[0] + self.player_collider_size[0]/2 

        for i in range(2):
            clip_bot = bot - self.ground_heights[i]
            clip_left = self.ground_rights[i] - left
            clip_right = right - self.ground_lefts[i]

            if clip_left < 0 or clip_right < 0 or clip_bot < -0.001:
                continue # no collision
            
            # collision; find direction to resolve collision with least movement
            if clip_bot < clip_left and clip_bot < clip_right: # move up to resolve
                # n_pos[1] -= max(0, clip_bot)
                # n_vel[1] = 0
                if n_vel[1] > 0: n_vel[1] = -n_vel[1] * self.player_bounce_coef
                on_ground = True
            else: # move horizontally to resolve
                if clip_left < clip_right: n_pos[0] += clip_left
                else: n_pos[0] -= clip_right
                n_vel[0] = 0

        #if not on_ground: print(self.env.steps)

        if (not on_ground) and apply_y_friction: # game doesn't update y_vel if on ground
            n_vel[1] += self.y_friction

        if not attacking: # their code caps fall speed in most states, but does not cap in attack state
            n_vel[1] = min(self.max_fall_speed, n_vel[1]) # game caps fall speed

        return n_pos, n_vel
      

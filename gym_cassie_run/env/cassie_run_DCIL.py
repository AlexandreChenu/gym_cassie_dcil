
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append(os.getcwd())

from abc import ABC
from abc import abstractmethod
from typing import Optional

# from humanoidenv import HumanoidEnv
from .cassie_run import CassieRunEnv

from typing import Union
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc

from collections import defaultdict, namedtuple

import copy
# from IPython import embed

# from .skill_manager_fetchenv import SkillsManager
# from skill_manager_fetchenv import SkillsManager

import gym
gym._gym_disable_underscore_compat = True

import types
os.environ["PATH"] = os.environ["PATH"].replace('/usr/local/nvidia/bin', '')
# try:
import mujoco_py

from gym.envs.mujoco import mujoco_env
# except Exception:
	# print('WARNING: could not import mujoco_py. This means robotics environments will not work')
import gym.spaces
from scipy.spatial.transform import Rotation
from collections import defaultdict, namedtuple
import os
from gym.envs.mujoco import mujoco_env


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


class GoalEnv(gym.Env):
	"""The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		super().reset(seed=seed)
		# Enforce that each GoalEnv uses a Goal-compatible observation space.
		if not isinstance(self.observation_space, gym.spaces.Dict):
			raise error.Error(
				"GoalEnv requires an observation space of type gym.spaces.Dict"
			)
		for key in ["observation", "achieved_goal", "desired_goal"]:
			if key not in self.observation_space.spaces:
				raise error.Error('GoalEnv requires the "{}" key.'.format(key))

	@abstractmethod
	def compute_reward(self, achieved_goal, desired_goal, info):
		"""Compute the step reward.
		Args:
			achieved_goal (object): the goal that was achieved during execution
			desired_goal (object): the desired goal
			info (dict): an info dictionary with additional information
		Returns:
			float: The reward that corresponds to the provided achieved goal w.r.t. to
			the desired goal. Note that the following should always hold true:
				ob, reward, done, info = env.step()
				assert reward == env.compute_reward(ob['achieved_goal'],
													ob['desired_goal'], info)
		"""
		raise NotImplementedError


@torch.no_grad()
def goal_distance(goal_a, goal_b):
	# assert goal_a.shape == goal_b.shape
	#print("\ngoal_a = ", goal_a)
	#print("goal_b = ", goal_b)
	#print("d = ", np.linalg.norm(goal_a - goal_b, axis=-1))
	if torch.is_tensor(goal_a):
		return torch.linalg.norm(goal_a - goal_b, axis=-1)
	else:
		return np.linalg.norm(goal_a - goal_b, axis=-1)


@torch.no_grad()
def default_compute_reward(
		achieved_goal: Union[np.ndarray, torch.Tensor],
		desired_goal: Union[np.ndarray, torch.Tensor],
		info: dict
):
	distance_threshold = 0.075
	reward_type = "sparse"
	d = goal_distance(achieved_goal, desired_goal)
	if reward_type == "sparse":
		# if torch.is_tensor(achieved_goal):
		#     return (d < distance_threshold).double()
		# else:
		return 1.0 * (d <= distance_threshold)
	else:
		return -d

## interface vers Humanoid
class GCassie(mujoco_env.MujocoEnv, utils.EzPickle, ABC):
	TARGET_SHAPE = 0
	MAX_PIX_VALUE = 0

	def __init__(self):

		self.env = CassieRunEnv(
		)

		self.action_space = self.env.action_space

		self.observation_space = self.env.observation_space

		self.set_reward_function(default_compute_reward)

		init_state = self.env.reset()
		self.init_state = init_state.copy()

		self.init_sim_state = self.env.get_inner_state()
		self.init_qpos = self.init_sim_state[0].copy()
		self.init_qvel = self.init_sim_state[1].copy()

		self.render_cache = defaultdict(dict)

		self.state = self.init_state.copy()
		self.done = False
		self.steps = 0

		self.max_episode_steps = 100

		self.rooms = []

		# self.viewer = mujoco_py.MjViewer(self.env.sim)

	def __getattr__(self, e):
		assert self.env is not self
		return getattr(self.env, e)

	def set_reward_function(self, reward_function):
		self.compute_reward = (
			reward_function  # the reward function is not defined by the environment
		)


	def reset_model(self, env_indices=None) -> np.ndarray:
		"""
		Reset environments to initial simulation state & return vector state
		"""
		self.env.set_inner_state(self.init_sim_state)

		return self.state_vector()

	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		self.reset_model()
		self.steps = 0
		return self.state_vector()

	def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
		self.reset_model()
		self.steps = 0
		return self.state_vector()

	def step(self, action):
		self.steps += 1
		cur_state = self.state.copy()

		new_state, env_reward, done, _, info =  self.env.step(action)
		# print("step : ", self.project_to_goal_space(new_state))
		self.state = new_state
		reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, {})

		truncation = (self.steps >= self.max_episode_steps)

		is_success = reward.copy().reshape(1,)
		self.is_success = is_success.copy()

		truncation = truncation * (1 - is_success).reshape(1,)
		info = {'is_success': is_success,
				'done_from_env': np.array(done,dtype=np.intc).reshape(1,),
				'reward_from_env': env_reward,
				'truncation': truncation}
		self.done = (done or bool(truncation)) or bool(is_success)
		return self.state_vector(), reward, self.done, info


	def state_vector(self):
		return self.env._get_obs().copy()

	def render(self):
		return self.env.render()


	def set_state(self, sim_state, set_state):
		if set_state:
			self.env.set_inner_state(sim_state)
			self.state = self.env._get_obs().copy()

	def get_state(self):
		state = (self.state.copy(), self.env.get_inner_state())
		return state

	def get_observation(self):
		return self.state.copy()

	def set_goal(self, goal, set_goal):
		if set_goal:
			self.goal = goal.copy()

	def get_goal(self):
		return self.goal.copy()

	def set_max_episode_steps(self, max_episode_steps, set_steps):
		if set_steps:
			self.max_episode_steps = max_episode_steps
			self.steps = 0

	def get_max_episode_steps(self):
		return self.max_episode_steps

	def get_obs_dim(self):
		return self.get_state()[0].shape[0]

	@torch.no_grad()
	def project_to_goal_space(self, state):
		com_pos = self.get_com_pos(state)

		return com_pos

	def get_com_pos(self, state):
		"""
		get center of mass position from full state for torso?
		"""
		# print("len(list(state)) = ", len(list(state)))
		# assert len(list(state))== 378

		com_pos = state[:3]
		# gripper_pos = state[102:105]

		return com_pos


@torch.no_grad()
def goal_distance(goal_a, goal_b):
	# assert goal_a.shape == goal_b.shape
	#print("\ngoal_a = ", goal_a)
	#print("goal_b = ", goal_b)
	#print("d = ", np.linalg.norm(goal_a - goal_b, axis=-1))
	if torch.is_tensor(goal_a):
		return torch.linalg.norm(goal_a - goal_b, axis=-1)
	else:
		return np.linalg.norm(goal_a - goal_b, axis=-1)


@torch.no_grad()
def default_compute_reward(
		achieved_goal: Union[np.ndarray, torch.Tensor],
		desired_goal: Union[np.ndarray, torch.Tensor],
		info: dict
):
	distance_threshold = 0.05
	reward_type = "sparse"
	d = goal_distance(achieved_goal, desired_goal)
	if reward_type == "sparse":
		# if torch.is_tensor(achieved_goal):
		#     return (d < distance_threshold).double()
		# else:
		return 1.0 * (d <= distance_threshold)
	else:
		return -d

class GCassieGoal(GCassie, GoalEnv, utils.EzPickle, ABC):
	def __init__(self):
		super().__init__()

		self.reset_model()
		self._goal_dim = self.project_to_goal_space(self.state).shape[0] ## TODO: set automatically
		high_goal = np.ones(self._goal_dim)
		low_goal = -high_goal

		self.observation_space = spaces.Dict(
			dict(
				observation=self.env.observation_space,
				achieved_goal=spaces.Box(
					low_goal, high_goal, dtype=np.float64
				),
				desired_goal=spaces.Box(
					low_goal, high_goal, dtype=np.float64
				),
			)
		)

		self.goal = None

		self.compute_reward = None
		self.set_reward_function(default_compute_reward)

		self._is_success = None
		# self.set_success_function(default_success_function)

	def get_obs_dim(self):
		return self.get_state()[0].shape[0]


	def get_goal_dim(self):
		return self._goal_dim

	@torch.no_grad()
	def goal_distance(self, goal_a, goal_b):
		# assert goal_a.shape == goal_b.shape
		if torch.is_tensor(goal_a):
			return torch.linalg.norm(goal_a - goal_b, axis=-1)
		else:
			return np.linalg.norm(goal_a - goal_b, axis=-1)


	@torch.no_grad()
	def step(self, action):
		self.steps += 1
		cur_state = self.state.copy()

		new_state, env_reward, done, _, info =  self.env.step(action)
		# print("step : ", self.project_to_goal_space(new_state))
		self.state = new_state
		reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, {})

		truncation = (self.steps >= self.max_episode_steps)

		is_success = reward.copy().reshape(1,)
		self.is_success = is_success.copy()

		truncation = truncation * (1 - is_success).reshape(1,)
		info = {'is_success': is_success,
				'done_from_env': np.array(done,dtype=np.intc).reshape(1,),
				'reward_from_env': env_reward,
				'truncation': truncation}
		self.done = (done or bool(truncation)) or bool(is_success)

		# print("observation env = ", self.state[:15])

		return (
			{
				'observation': self.state.copy(),
				'achieved_goal': self.project_to_goal_space(self.state),
				'desired_goal': self.goal.copy(),
			},
			reward,
			self.done,
			info,
		)

	@torch.no_grad()
	def _sample_goal(self):
		# return (torch.rand(self.num_envs, 2) * 2. - 1).to(self.device)
		return np.random.uniform(-1.,1., size=self._goal_dim)

	@torch.no_grad()
	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		self.reset_model()  # reset state to initial value
		self.goal = self._sample_goal()  # sample goal
		self.steps = 0
		self.state = self.state_vector()
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}

	def reset_done(self, options=None, seed: Optional[int] = None, infos=None):

		# self.reset_model() ## do not force reset model if overshoot used
		self.goal = self._sample_goal()
		self.steps = 0.
		self.state = self.state_vector()

		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}

	@torch.no_grad()
	def project_to_goal_space(self, state):
		com_pos = self.get_com_pos(state)

		return com_pos

	def get_com_pos(self, state):
		"""
		get center of mass position from full state for torso?
		"""
		# print("len(list(state)) = ", len(list(state)))
		# assert len(list(state))== 378

		com_pos = state[:3]
		# gripper_pos = state[102:105]

		return com_pos

	def set_state(self, sim_state, set_state):
		if set_state:
			self.env.set_inner_state(sim_state)
			self.state = self.env._get_obs().copy()

	def get_state(self):
		state = (self.env._get_obs().copy(), self.env.get_inner_state())
		return state

	def get_observation(self):
		return {
			'observation': self.env._get_obs().copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}

	def set_goal(self, goal, set_goal):
		if set_goal:
			self.goal = goal.copy()

	def get_goal(self):
		return self.goal.copy()

	def set_max_episode_steps(self, max_episode_steps, set_steps):
		if set_steps:
			self.max_episode_steps = max_episode_steps
			self.steps = 0

	def get_max_episode_steps(self):
		return self.max_episode_steps


if (__name__=='__main__'):

	env = GCassieGoal()

	obs = env.reset()
	print("obs = ", obs)

	for i in range(100):
		env.env.render()
		action = env.action_space.sample()
		print("sim_state = ", env.sim.get_state())
		obs, reward, done, info = env.step(action)

	obs = env.reset()
	print("obs = ", obs)

	for i in range(100):
		env.env.render()
		action = env.action_space.sample()
		print("sim_state = ", env.sim.get_state())
		obs, reward, done, info = env.step(action)

		# if done:
			# env.reset_done()

	print("obs = ", obs)
	print("done = ", done)
	print("info = ", info)

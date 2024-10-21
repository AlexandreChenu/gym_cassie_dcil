import sys
import os
sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from os import path

from .cassie_run import *

import gymnasium as gym

from gymnasium import utils, spaces, error
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
	"distance": 4.0,
}



class GoalEnv(gym.Env):
	"""The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

	def reset(self, options=None, seed: Optional[int] = None):
		super().reset(seed=seed)
		# Enforce that each GoalEnv uses a Goal-compatible observation space.
		if not isinstance(self.observation_space, gym.spaces.Dict):
			raise error.Error(
				"GoalEnv requires an observation space of type gym.spaces.Dict"
			)
		for key in ["observation", "achieved_goal", "desired_goal"]:
			if key not in self.observation_space.spaces:
				raise error.Error('GoalEnv requires the "{}" key.'.format(key))

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


def goal_distance(goal_a, goal_b):
	# assert goal_a.shape[1] == 2
	# assert goal_b.shape[1] == 2
	#print("goal_a.shape = ", goal_a.shape)
	#print("goal_b.shape = ", goal_b.shape)

	return np.linalg.norm(goal_a[:] - goal_b[:], axis=-1)


def default_compute_reward(
		achieved_goal: np.ndarray,
		desired_goal: np.ndarray):
	
	distance_threshold = 0.1
	reward_type = "sparse"
	d = goal_distance(achieved_goal, desired_goal)
	if reward_type == "sparse":
		return 1.0 * (d <= distance_threshold)
	else:
		return -d

def default_success_function(achieved_goal, desired_goal):
	distance_threshold = 0.1
	d = goal_distance(achieved_goal, desired_goal)
	return 1.0 * (d <= distance_threshold)



class GCassieRunEnv(CassieRunEnv, GoalEnv, utils.EzPickle, ABC):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		
		self._obs_dim = (67,)
		
		self.max_episode_steps = 1000

		high = np.ones(self._obs_dim)
		low = -high
		self._achieved_goal_dim = 3
		self._desired_goal_dim = 3
		high_achieved_goal = np.ones(self._achieved_goal_dim)
		low_achieved_goal = -high_achieved_goal
		high_desired_goal = np.ones(self._desired_goal_dim)
		low_desired_goal = -high_desired_goal
		self.observation_space = spaces.Dict(
			dict(
				observation=spaces.Box(low, high, dtype=np.float64),
				achieved_goal=spaces.Box(
					low_achieved_goal, high_achieved_goal, dtype=np.float64
				),
				desired_goal=spaces.Box(
					low_desired_goal, high_desired_goal, dtype=np.float64
				),
			)
		)
				
		self.goal = None

		self.compute_reward = None
		self.set_reward_function(default_compute_reward)

		self._is_success = None

		# print("self.max_episode_steps.shape = ", self.max_episode_steps.shape)
		# self.set_success_function(default_success_function)

	def project_to_goal_space(self, state):
		return state[:3]

	def get_obs_dim(self):
		return self._obs_dim
	def get_full_state_dim(self):
		return self._obs_dim
	def get_goal_dim(self):
		return self._achieved_goal_dim

	def goal_distance(self, goal_a, goal_b):
		# assert goal_a.shape == goal_b.shape
		return np.linalg.norm(goal_a[:] - goal_b[:], axis=-1)

	def set_reward_function(self, reward_function):
		self.compute_reward = (  # the name is compute_reward in GoalEnv environments
			reward_function
		)


	def reset(self, *, options=None, seed: Optional[int] = None):
		self.reset_model()
		self.goal = self._sample_goal()  # sample goal
		self.steps = 0
		self._elapsed_steps = 0
		self.state = self._get_obs()

		info = {}

		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}, info

	# TODO adapt to vec env
	def reset_done(self, done, *, options=None, seed: Optional[int] = None):
		if done : 
			self.steps = 0
			self._elapsed_steps = 0
			newgoal = self._sample_goal()
			self.goal = newgoal.copy()
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}, {}

	def set_state_(self, state, new_state_bool=1):
		state = state.flatten()
		if new_state_bool:
			new_qpos = state[:self.init_qpos.shape[0]]
			new_qvel = state[self.init_qpos.shape[0]:]
			self.set_state(new_qpos, new_qvel)

			self.state = self._get_obs()
		return self.get_state()

	def get_state(self):
		return self._get_obs()

	def get_observation(self):
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}

	def set_goal(self, goal, new_goal_bool):
		if new_goal_bool:
			self.goal = goal.copy().reshape(-1,)

	def get_goal(self,):
		return self.goal

	def set_max_episode_steps(self, max_episode_steps, new_max_episode_steps_bool):
		if new_max_episode_steps_bool:
			self.max_episode_steps = max_episode_steps


	def _sample_goal(self):
		return self.project_to_goal_space(np.random.rand(3) * 100 )
	
	def step(self, action):
		action = action.flatten()
		x_position_before = (
			2.0 * self.data.qpos[0]
			+ self.data.body("left-foot").xpos[0]
			+ self.data.body("right-foot").xpos[0]
		)
		true_action = (action + 1.0) / 2.0 * (
			self._action_upper_bounds - self._action_lower_bounds
		) + self._action_lower_bounds
		self.do_simulation(true_action, self.frame_skip)

		self.steps += 1

		x_position_after = (
			2.0 * self.data.qpos[0]
			+ self.data.body("left-foot").xpos[0]
			+ self.data.body("right-foot").xpos[0]
		)
		x_velocity = 2.0 * (x_position_after - x_position_before) / self.dt
		z_position_after = self.data.qpos[2]

		ctrl_cost = self.control_cost(true_action)

		forward_reward = (
			self._forward_reward_weight * x_velocity
			- np.abs(self.data.qpos[1]) * 10.0
			- self.quat_distance(
				self.init_foot_quat, self.data.body("right-foot").xquat
			)
			* 5
			- self.quat_distance(self.init_foot_quat, self.data.body("left-foot").xquat)
			* 5
		)
		self.state = self._get_obs()
		terminated = False

		reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal)
		reward = np.array(reward).reshape(1,)

		is_success = reward.copy()

		done = np.zeros(is_success.shape)
		terminated = done.copy()

		truncation = np.array((self.steps >= self.max_episode_steps)).astype(int).reshape(1,)

		info = {
			"x_position": x_position_after,
			"x_velocity": x_velocity,
			"reward_run": forward_reward,
			"reward_ctrl": -ctrl_cost,
			'is_success': is_success,
			'done_from_env': done,
			'truncation': truncation,
		}

		self.done = np.maximum(truncation, is_success)

		if self.render_mode == "human":
			self.render()
			
		return (
			{
				'observation': self.state.copy(),
				'achieved_goal': self.project_to_goal_space(self.state),
				'desired_goal': self.goal.copy(),
			},
			reward,
			terminated, 
			truncation, 
			info,
		)

if (__name__=='__main__'):

	env = GCassieRunEnv(render_mode = "human")

	obs, info = env.reset()
	# print("obs = ", obs)

	for i in range(200):
		
		action = env.action_space.sample()
		# print("sim_state = ", env.sim.get_state())
		env.step(action)

		env.render()


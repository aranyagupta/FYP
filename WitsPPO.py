import kan
import torch
from kan.utils import create_dataset
import matplotlib.pyplot as plt
from kan.utils import ex_round
from sklearn import datasets
import numpy as np
import time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""


# PPO for full Witsenhausen counterexample
torch.set_default_dtype(torch.float32)

class WitsGradDesc:
	def __init__(self, env, actor_c1, actor_c2, noise=True):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 0.01
		self.noise = noise

		self.actor_c1_optim = Adam(self.actor_c1.parameters(), lr=self.lr)
		self.actor_c2_optim = Adam(self.actor_c2.parameters(), lr=self.lr)

		# self.actor_c1_optim = torch.optim.SGD(self.actor_c1.parameters(), lr=self.lr, momentum=0.5, nesterov=True)
		# self.actor_c2_optim = torch.optim.SGD(self.actor_c1.parameters(), lr=self.lr, momentum=0.5, nesterov=True)

	def train(self, timesteps, batches):
		start = time.time()
		for batch in range(batches):
			# print("batch:", batch)
			_, _, reward, _, _ = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps, noise=self.noise)
			reward = -reward
			
			actor_loss = reward.mean()
			print("DGD LOSS:",actor_loss)

			self.actor_c1_optim.zero_grad()
			self.actor_c2_optim.zero_grad()
			actor_loss.backward()
			self.actor_c2_optim.step()
			self.actor_c1_optim.step()
		print("TOOK", time.time()-start, "seconds to finish.")
			
class WitsGradDescConstrained:
	def __init__(self, env, actor_c1, actor_c2, noise=True):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 0.01
		self.noise = noise

		self.lamb_0 = torch.tensor(1.0, requires_grad=True, device = actor_c1.device)
		self.lamb_1 = torch.tensor(1.0, requires_grad=True, device = actor_c1.device)
		self.mu_0 = torch.tensor(1.0, requires_grad=True, device=actor_c1.device)
		self.mu_1 = torch.tensor(1.0, requires_grad=True, device=actor_c1.device)

		self.actor_c1_optim = Adam(self.actor_c1.parameters(), lr=self.lr)
		self.actor_c2_optim = Adam(self.actor_c2.parameters(), lr=self.lr)
		self.mu_0_optim = Adam([self.mu_0], lr=self.lr, maximize=True) # gradient ascent for langrangian multipliers
		self.mu_1_optim = Adam([self.mu_1], lr=self.lr, maximize=True)
		self.lamb_0_optim = Adam([self.lamb_0], lr=self.lr, maximize=True)
		self.lamb_1_optim = Adam([self.lamb_1], lr=self.lr, maximize=True)

	def train(self, timesteps, batches):
		start = time.time()
		for batch in range(batches):
			# print("batch:", batch)
			_, _, reward, _, _ = self.env.step_timesteps(self.actor_c1, self.actor_c2, self.lamb_0, self.lamb_1, self.mu_0, self.mu_1, timesteps, noise=self.noise)
			
			actor_loss = reward.mean()
			print("LAG LOSS:",actor_loss)
			print("lambda:", self.lamb_0.item(), self.lamb_1.item())
			print("mu:", self.mu_0.item(), self.mu_1.item())

			self.actor_c1_optim.zero_grad()
			self.actor_c2_optim.zero_grad()
			self.mu_0_optim.zero_grad()
			self.mu_1_optim.zero_grad()
			self.lamb_0_optim.zero_grad()
			self.lamb_1_optim.zero_grad()

			actor_loss.backward()
			
			self.mu_0_optim.step()
			self.mu_1_optim.step()
			self.lamb_0_optim.step()
			self.lamb_1_optim.step()

			# clamp mu multipliers to be >=0 (as they are inequality constraints)
			with torch.no_grad():
				self.mu_0.clamp_(min=0)
				self.mu_1.clamp_(min=0)
			
			# gradient descent for model (minimise J)
			self.actor_c2_optim.step()
			self.actor_c1_optim.step()
		print("TOOK", time.time()-start, "seconds to finish.")

class WitsPPOCombined:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, env, actor_combined, actor_combined2, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = 1
		self.act_dim = 1

		 # Initialize actor and critic networks
		self.actor = actor_combined                                           # ALG STEP 1
		self.critic = kan.KAN()
		self.critic.initialize_from_another_model(actor_combined)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		

	def train(self, timesteps, batches):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				batches - the number of batches to train for
				timesteps - the number of timesteps to train for

			Return:
				None
		"""
		total_timesteps = batches*timesteps
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts, batch_rtgs)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient descent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward()
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()
		
	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []


		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			observations_c1, _, rewards, _, _ = self.env.step_timesteps(self.max_timesteps_per_episode, self.actor)
			batch_obs.append(observations_c1)	

			actions, log_probs = self.get_action(observations_c1)
			batch_rews.append(rewards)
			batch_acts.append(actions)
			batch_log_probs.append(log_probs)

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			

			# Track episodic lengths and rewards
			batch_lens.append(self.max_timesteps_per_episode)
			t += self.max_timesteps_per_episode

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.stack(batch_obs).reshape(self.timesteps_per_batch, self.obs_dim)
		batch_acts = torch.stack(batch_acts).reshape(self.timesteps_per_batch, self.act_dim)
		batch_log_probs = torch.stack(batch_log_probs).flatten()
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4


		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num episodes per batch, num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		_, _, mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# If we're testing, just return the deterministic action. Sampling should only be for training
		# as our "exploration" factor.
		if self.deterministic:
			return mean.detach(), 1

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts, batch_rtgs):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
				batch_rtgs - the rewards-to-go calculated in the most recently collected
								batch as a tensor. Shape: (number of timesteps in batch)
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		_, _, mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 1000                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 100            # Max number of timesteps per episode
		self.n_updates_per_iteration = 1                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = False                             # If we should render during rollout
		self.save_freq = 10                             # How often we save in number of iterations
		self.deterministic = False                      # If we're testing, don't sample actions
		self.seed = None								# Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")


class WitsGradDescCombined:
	def __init__(self, env, actor_combined, actor_combined2):
		self.env = env
		self.actor = actor_combined
		self.lr = 0.005

		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

	def train(self, timesteps, batches):
		for batch in range(batches):
			# print("batch:", batch)
			_, _, reward, _, _ = self.env.step_timesteps(timesteps, self.actor)
			reward = -reward
			
			actor_loss = reward.mean()
			print("LOSS:",actor_loss)

			self.actor_optim.zero_grad()
			actor_loss.backward()
			self.actor_optim.step()

class WitsAlternatingDescent:
	def __init__(self, env, actor_c1, actor_c2, noise=True):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 0.01
		self.noise = noise

		self.actor_c1_optim = Adam(self.actor_c1.parameters(), lr=self.lr)
		self.actor_c2_optim = Adam(self.actor_c2.parameters(), lr=self.lr)
	
	def train(self, timesteps, batches, n_repeats=1):
		for batch in range(batches):
			for n in range(n_repeats):
				_, _, reward, _, _ = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps, noise=self.noise)
				reward = -reward
				
				actor_loss = reward.mean()
				print("ALTERNATING LOSS:", actor_loss)

				self.actor_c2_optim.zero_grad()
				actor_loss.backward()
				self.actor_c2_optim.step()
			
			for n in range(n_repeats):
				_, _, reward, _, _ = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps, noise=self.noise)
				reward = -reward
				actor_loss = reward.mean()
				self.actor_c1_optim.zero_grad()
				actor_loss.backward()
				self.actor_c1_optim.step()
				print("ALTERNATING LOSS:", actor_loss)

# Local Search Algorithm as described in this paper:
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8264401	
class WitsLSA:
	def __init__(self, env, actor_c1, actor_c2=None, N=15, r=0.25, p=0.01):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 1e-2 # lr used for all parameters of model, keep low

		self.N = N # num repetitions
		self.r = r # local smoothing radius
		self.p = p # precision
		self.tau = 1e-3 # grad descent step size

		self.actor_c1_optim = torch.optim.Adam(self.actor_c1.parameters(), lr=self.lr)
		# SGD is a closer implementation to what we want to do

	def train(self, timesteps, batches):
		while True:
			for i in range(self.N):
				dJ_dx1, dx1_dJ_dx1, out, x_0 = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
				gradients = torch.zeros_like(dx1_dJ_dx1)
				count = 0
				gradients = dJ_dx1/torch.abs(dx1_dJ_dx1) 
				for i in range(dx1_dJ_dx1.shape[0]):
					if torch.abs(dx1_dJ_dx1[i]) <= 1e-6:
						gradients[i] = self.tau * dJ_dx1[i] # positive as optimiser automatically handles gradient descent
						count+=1
					else:
						gradients[i] = dJ_dx1[i]/torch.abs(dx1_dJ_dx1[i]) 
				print("gd count:", count)
				
				# print("gradients has nan:", torch.any(torch.isnan(gradients)))
				self.actor_c1_optim.zero_grad()
				out.backward(gradient=gradients, retain_graph=True)
				self.actor_c1_optim.step()

			# Skip smoothing step as it doesn't work nicely with KANs (doesn't produce clean update law)

			x_0_integrating, indices = torch.sort(x_0, dim=0)
			stop_condition = torch.trapz(torch.abs(dJ_dx1[indices].reshape(dJ_dx1.shape[0], 1)), x_0_integrating, dim=0)
			print("STOP CONDITION:", stop_condition)
			if stop_condition < self.p:
				break
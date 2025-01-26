import kan
import torch
from kan.utils import create_dataset
import matplotlib.pyplot as plt
from kan.utils import ex_round
import numpy as np
import WitsEnv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal



"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""


class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, actor, critic, env, **hyperparameters):
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
		self.actor = actor                                           # ALG STEP 1
		self.critic = critic

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
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

				print("ACTOR LOSS:", actor_loss)

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
			observations, rewards, _, _ = self.env.step_timesteps(self.actor, self.max_timesteps_per_episode)
			batch_obs.append(observations)	

			actions, log_probs = self.get_action(observations)
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
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float64)

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
		mean = self.actor(obs)

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
		mean = self.actor(batch_obs)
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


class SimpleGradDesc:
	def __init__(self, actor, env, device):
		self.device = device
		self.actor = actor
		self.env = env
		self.lr = 0.005

		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

	def train(self, timesteps, batches, requires_grad=False):
		for batch in range(batches):
			print("Batch:", batch)
			_, rews, _, _ = self.env.step_timesteps(self.actor, timesteps)
			rews = -rews
			
			
			actor_loss = rews.mean()
			if requires_grad:
				actor_loss.requires_grad = True
			# Calculate gradients and perform backward propagation for actor network
			self.actor_optim.zero_grad()
			actor_loss.backward()
			self.actor_optim.step()
			print(actor_loss)


			# self.actor.plot()
			# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_device(device=device)
torch.set_default_dtype(torch.float64)


# ---------------- SIMPLE GRAD DESC --------------- #
# simple backprop
sigma=5.0
grid_range = [-3*sigma, 3*sigma]
actor = kan.KAN(width=[1, 2, 2, 1], grid=11, k=5, seed=42, grid_range=grid_range, device=device)
critic = kan.KAN(width=[1, 2, 2, 1], grid=7, k=5, seed=42, grid_range=grid_range, device=device)
env = WitsEnv.WitsEnvSimple(k=0.2, sigma=5, actor_c1=actor, dims=1, device=device)
simple = SimpleGradDesc(actor, env, device)
simple.train(100, 1000)
actor.plot()
plt.show()

# Prune
actor.prune()
actor.plot()
plt.show()

# Train
simple.train(1000, 500)
actor.plot()
plt.show()

# Auto-symbollise 
actor.auto_symbolic()
actor.plot()
plt.show()

# Train
simple.train(1000, 1000)
actor.plot()
plt.show()


# Train
simple.train(1000, 500)
actor.plot()
plt.show()

# Train
simple.train(1000, 500)
actor.plot()
plt.show()

# Auto-symbollise 
actor.auto_symbolic()
actor.plot()
plt.show()

# Train
simple.train(1000, 1000)
actor.plot()
plt.show()

actor.prune()
actor.plot()
plt.show()

actor.symbolic_formula()[0][0]
form = ex_round(actor.symbolic_formula()[0][0],4)
print(form)
actor.plot()
plt.show()



# -------------------------- PPO -------------------- #
sigma = 5.0
grid_range = [-6*sigma, 6*sigma]
actor = kan.KAN(width=[1, 2, 2, 1], grid=3, k=5, seed=42, grid_range=grid_range, device=device)
critic = kan.KAN(width=[1, 2, 2, 1], grid=3, k=5, seed=42, grid_range=grid_range, device=device)
env = WitsEnv.WitsEnvSimple(k=0.2, sigma=5, actor_c1=actor, dims=1, device=device)
ppo = PPO(actor, critic, env)

# Train
# actor.plot()
# plt.show()
test = WitsEnv.WitsActorTestSimple(actor, device)
# print("INITIAL TEST: ", test.test(1000))
# actor.plot()
# plt.show()

# learning over 10 batches, 20 episodes per batch and 100 timesteps per episode

ppo.learn(20000)

print("TEST: ", test.test(10000))
actor.plot()
plt.show()
critic.plot()
plt.show()


# Prune
actor.prune()
actor.plot()
plt.show()

# Train
ppo.learn(20000)
print("TEST: ", test.test(10000))
actor.plot()
plt.show()

# Refine
actor.refine(5)
actor.plot()
plt.show()

# Train
ppo.learn(20000)
print("TEST: ", test.test(10000))
actor.plot()
plt.show()

# Refine
actor.refine(7)
actor.plot()
plt.show()

# Train
ppo.learn(20000)
print("TEST: ", test.test(10000))
actor.plot()
plt.show()

# Auto-symbollise 
actor.auto_symbolic()
actor.plot()
plt.show()

# Train
ppo.learn(20000)
print("TEST: ", test.test(10000))
actor.plot()
plt.show()

# Final
print("FINAL:")
actor.prune()
actor.plot()
plt.show()

actor.symbolic_formula()[0][0]
form = ex_round(actor.symbolic_formula()[0][0],4)
print(form)
actor.plot()
plt.show()


# ----------------------------------------------------------------------------------------------------------- #

# # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = kan.KAN(width=[4,1,1], grid=3, k=5, seed=42, device=device)
# dataset = datasets.load_iris()

# n = len(dataset.target)
# perm = [x for x in range(n)]
# np.random.shuffle(perm)

# x = dataset.data[perm]
# y = dataset.target[perm]

# x_train = torch.tensor(x[:int(0.8*n)], device=device)
# y_train = torch.tensor(y[:int(0.8*n)], device=device).reshape(-1, 1)

# x_test = torch.tensor(x[int(0.8*n):], device=device)
# y_test = torch.tensor(y[int(0.8*n):], device=device).reshape(-1, 1)

# dataset = {
#     "train_input": x_train,
#     "train_label": y_train,
#     "test_input": x_test,
#     "test_label": y_test
# }

# # data shapes
# print(dataset["train_input"].shape)
# print(dataset["train_label"].shape)
# print(dataset["test_input"].shape)
# print(dataset["test_label"].shape)


# # untrained model
# model(dataset['train_input'])
# model.plot()
# plt.show()


# # model after 50 steps
# model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
# model.plot()
# plt.show()



# # prune model
# model = model.prune()
# model.plot()
# plt.show()

# # refined model
# model.fit(dataset, opt="LBFGS", steps=50)
# model = model.refine(5)
# model.plot()
# plt.show()

# model.fit(dataset, opt="LBFGS", steps=50)

# # activation functions set
# lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
# model.auto_symbolic(lib=lib)
# model.plot()
# plt.show()

# # continue training
# model.fit(dataset, opt="LBFGS", steps=50)

# # final model and solution
# form = ex_round(model.symbolic_formula()[0][0],4)
# print(form)
# model.plot()
# plt.show()

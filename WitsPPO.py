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
	def __init__(self, env, actor_c1, actor_c2=None, N=15, r=0.25, p=1e-3):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 1e-2 # lr used for all parameters of model, keep low

		self.N = N # num repetitions
		self.r = r # local smoothing radius
		self.p = p # precision
		self.tau = 1e-3 # grad descent step size

		self.actor_c1_optim = torch.optim.Adam(self.actor_c1.parameters(), lr=self.lr)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.actor_c1_optim, step_size=90, gamma=0.9)
		
	def train(self, timesteps, batches=0):
		while True:
			for i in range(self.N):
				dJ_dx1, dx1_dJ_dx1, out, x_0 = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
				gradients = dJ_dx1/torch.abs(dx1_dJ_dx1)
				small_mask = torch.abs(dx1_dJ_dx1) <= 1e-6
				gradients[small_mask] = self.tau * dJ_dx1[small_mask]
				########## FOR LOOP IMPLEMENTATION - SLOW ####################
				# for i in range(dx1_dJ_dx1.shape[0]):
				# 	if torch.abs(dx1_dJ_dx1[i]) <= 1e-6:
				# 		gradients[i] = self.tau * dJ_dx1[i] # positive as optimiser automatically handles gradient descent
				# 		count+=1
				########## FOR LOOP IMPLEMENTATION - SLOW ####################
				
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
			self.scheduler.step()

# Frechet Gradient Descent (NOT Frechet Discrete Gradient) training wrapper
class WitsFGD:
	def __init__(self, env, actor_c1, actor_c2):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = 1e-3 # small learning rate for model params
		self.tau = 1e-3 # tau as defined in paper, scaling factor for gradient

		self.actor_c1_optim = Adam(self.actor_c1.parameters(), lr=self.lr)
		self.actor_c2_optim = Adam(self.actor_c2.parameters(), lr=self.lr)

	def train(self, timesteps, batches):
		# print("batch:", batch)
		for batch in range(batches):
			gradient_1, gradient_2, out_1, out_2, J = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps=timesteps)
			
			self.actor_c1_optim.zero_grad()
			self.actor_c2_optim.zero_grad()

			out_1.backward(gradient=self.tau*gradient_1, retain_graph=True)
			out_2.backward(gradient=self.tau*gradient_2, retain_graph=True)

			self.actor_c2_optim.step()
			self.actor_c1_optim.step()

			print("PERFORMANCE:", J)
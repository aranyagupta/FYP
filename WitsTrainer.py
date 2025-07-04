import torch
import time
import torch
from torch.optim import Adam

torch.set_default_dtype(torch.float32)

'''
Superclass for all Witsenhausen Trainers
Every Trainer should implement this class
'''
class WitsTrainer:
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		self.env = env
		self.actor_c1 = actor_c1
		self.actor_c2 = actor_c2
		self.lr = lr

		if actor_c1 is not None:
			self.actor_c1_optim = Adam(self.actor_c1.parameters(), lr=self.lr)
		if actor_c2 is not None:
			self.actor_c2_optim = Adam(self.actor_c2.parameters(), lr=self.lr)

	def train(self, timesteps, batches):
		raise Exception("Not Implemented!")
		

class WitsGradDesc(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		'''
		Unconstrained Optimisation Trainer
		Args:
			env (WitsEnv): Witsenhausen environment
			actor_c1 (kan.KAN): KAN model representing controller 1 
			actor_c2 (kan.KAN): KAN model representing controller 2
			lr (float): learning rate for optimiser
		'''
		super().__init__(env, actor_c1, actor_c2, lr)

	def train(self, timesteps, batches):
		'''
		Perform backprop for model
		Args:
			timesteps (int): Number of timesteps per batch
			batch (int): number of batches
		'''
		start = time.time()
		for batch in range(batches):
			# print("batch:", batch)
			reward = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
			
			actor_loss = reward.mean()
			print("DGD LOSS:",actor_loss)

			self.actor_c1_optim.zero_grad()
			self.actor_c2_optim.zero_grad()
			actor_loss.backward()
			self.actor_c2_optim.step()
			self.actor_c1_optim.step()
		print("TOOK", time.time()-start, "seconds to finish.")

'''
Constrained Optimisation Trainer
'''
class WitsGradDescConstrained(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		'''
		Constrained Optimisation Trainer
		Args:
			env (WitsEnv): Witsenhausen environment
			actor_c1 (kan.KAN): KAN model representing controller 1 
			actor_c2 (kan.KAN): KAN model representing controller 2
			lr (float): learning rate for optimiser
		'''
		super().__init__(env, actor_c1, actor_c2, lr)

		self.lamb_0 = torch.tensor(1.0, requires_grad=True, device = actor_c1.device)
		self.lamb_1 = torch.tensor(1.0, requires_grad=True, device = actor_c1.device)
		self.mu_0 = torch.tensor(1.0, requires_grad=True, device=actor_c1.device)
		self.mu_1 = torch.tensor(1.0, requires_grad=True, device=actor_c1.device)

		self.mu_0_optim = Adam([self.mu_0], lr=self.lr, maximize=True) # gradient ascent for langrangian multipliers
		self.mu_1_optim = Adam([self.mu_1], lr=self.lr, maximize=True)
		self.lamb_0_optim = Adam([self.lamb_0], lr=self.lr, maximize=True)
		self.lamb_1_optim = Adam([self.lamb_1], lr=self.lr, maximize=True)

	def train(self, timesteps, batches):
		'''
		Perform backprop for model
		Args:
			timesteps (int): Number of timesteps per batch
			batch (int): number of batches
		'''
		start = time.time()
		for batch in range(batches):
			# print("batch:", batch)
			reward = self.env.step_timesteps(self.actor_c1, self.actor_c2, self.lamb_0, self.lamb_1, self.mu_0, self.mu_1, timesteps)
			
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

'''
"Combined" Gradient Descent Trainer (Combined Actor)
Deprecated, do not use
'''
class WitsGradDescCombined(WitsTrainer):
	def __init__(self, env, actor_combined, actor_combined2=None, lr=0.01):
		super().__init__(env, actor_combined, actor_combined2, lr)
		
	def train(self, timesteps, batches):
		for batch in range(batches):
			# print("batch:", batch)
			reward = self.env.step_timesteps(timesteps, self.actor_c1)
			
			actor_loss = reward.mean()
			print("LOSS:",actor_loss)

			self.actor_c1_optim.zero_grad()
			actor_loss.backward()
			self.actor_c1_optim.step()
'''
"Alternating" Gradient Descent Trainer (coordinate direction update)
Deprecated, do not use
'''
class WitsAlternatingDescent(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		super().__init__(env, actor_c1, actor_c2, lr)
	
	def train(self, timesteps, batches, n_repeats=1):
		for batch in range(batches):
			for n in range(n_repeats):
				reward = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
				actor_loss = reward.mean()

				self.actor_c2_optim.zero_grad()
				actor_loss.backward()
				self.actor_c2_optim.step()
				print("ALTERNATING LOSS:", actor_loss)
			
			for n in range(n_repeats):
				reward = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
				actor_loss = reward.mean()
				
				self.actor_c1_optim.zero_grad()
				actor_loss.backward()
				self.actor_c1_optim.step()
				print("ALTERNATING LOSS:", actor_loss)

'''
Local Search Algorithm Trainer as described in this paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8264401
batches and actor_c2 are irrelevant when training
Defaulted to None/0 respectively
'''
class WitsLSA(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2=None, lr=0.01, N=15, p=1e-3):
		'''
		Constrained Optimisation Trainer
		Args:
			env (WitsEnv): Witsenhausen environment
			actor_c1 (kan.KAN): KAN model representing controller 1 
			actor_c2 (kan.KAN): KAN model representing controller 2. Not used for LSA
			lr (float): learning rate for optimiser
			N (int): number of update repetitions before stop condition is checked
			p (float): stop condition precision
		'''
		super().__init__(env, actor_c1, actor_c2, lr)

		self.N = N # num repetitions
		self.p = p # precision
		self.tau = 1e-3 # grad descent step size

		self.scheduler = torch.optim.lr_scheduler.StepLR(self.actor_c1_optim, step_size=90, gamma=0.9)
		
	def train(self, timesteps, batches=0):
		'''
		Perform backprop for model
		Args:
			timesteps (int): Number of timesteps per batch
			batch (int): number of batches. Not used for LSA as stop condition is built-in.
		'''
		while True:
			for i in range(int(self.N)):
				dJ_dx1, dx1_dJ_dx1, out, x_0 = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps)
				gradients = dJ_dx1/torch.abs(dx1_dJ_dx1)
				small_mask = torch.abs(dx1_dJ_dx1) <= 1e-6
				gradients[small_mask] = self.tau * dJ_dx1[small_mask]
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

'''
Frechet Gradient Descent Trainer
'''
class WitsFGD(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		'''
		Initialised Frechet Gradient Descent Trainer
		Args:
			env (WitsEnv): Witsenhausen environment
			actor_c1 (kan.KAN): KAN model representing controller 1 
			actor_c2 (kan.KAN): KAN model representing controller 2. Not used for LSA
			lr (float): learning rate for optimiser
		'''
		super().__init__(env, actor_c1, actor_c2, lr)
		self.tau = 1e-3 # tau as defined in paper, scaling factor for gradient

	def train(self, timesteps, batches):
		'''
		Perform backprop for model
		Args:
			timesteps (int): Number of timesteps per batch
			batch (int): number of batches.
		'''
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

'''
Polyak Momentum + Frechet Gradient Trainer
'''
class WitsMomentum(WitsTrainer):
	def __init__(self, env, actor_c1, actor_c2, lr=0.01):
		'''
		Initialised Frechet Gradient Descent Trainer
		Args:
			env (WitsEnv): Witsenhausen environment
			actor_c1 (kan.KAN): KAN model representing controller 1 
			actor_c2 (kan.KAN): KAN model representing controller 2. Not used for LSA
			lr (float): learning rate for optimiser
		'''
		super().__init__(env, actor_c1, actor_c2, lr)
		self.tau = 1e-3 # tau as defined in paper, scaling factor for gradient

	def train(self, timesteps, batches):
		'''
			Perform backprop for model
			Args:
				timesteps (int): Number of timesteps per batch
				batch (int): number of batches.
		'''
		for batch in range(batches):
			
			# if batch < 50:
			# 	with torch.no_grad():
			# 		domain = torch.linspace(-15.0, 15.0, 1000, device=self.env.device).reshape(1000, 1)
			# 		m1_out = self.actor_c1(domain)
			# 		m2_out = self.actor_c2(domain)
			# 		torch.save(m1_out, f"batch-{batch}-m1.pt")
			# 		torch.save(m2_out, f"batch-{batch}-m2.pt")
			# if batch < 100:
			# 	if batch % 10 == 0 and self.env.k == 0.20 and self.env.sigma == 5.00:
			# 		with torch.no_grad():
			# 			domain = torch.linspace(-15.0, 15.0, 1000, device=self.env.device).reshape(1000, 1)
			# 			m1_out = self.actor_c1(domain)
			# 			m2_out = self.actor_c2(domain)
			# 			torch.save(m1_out, f"batch-{batch}-m1.pt")
			# 			torch.save(m2_out, f"batch-{batch}-m2.pt")
			# elif batch % 50 == 0 and self.env.k == 0.20 and self.env.sigma == 5.00:
			# 	with torch.no_grad():
			# 			domain = torch.linspace(-15.0, 15.0, 1000, device=self.env.device).reshape(1000, 1)
			# 			m1_out = self.actor_c1(domain)
			# 			m2_out = self.actor_c2(domain)
			# 			torch.save(m1_out, f"batch-{batch}-m1.pt")
			# 			torch.save(m2_out, f"batch-{batch}-m2.pt")
			gradient_1, gradient_2, out_1, out_2, J = self.env.step_timesteps(self.actor_c1, self.actor_c2, timesteps=timesteps)

			self.actor_c1_optim.zero_grad()
			self.actor_c2_optim.zero_grad()

			out_1.backward(gradient=self.tau*gradient_1, retain_graph=False)
			out_2.backward(gradient=self.tau*gradient_2, retain_graph=False)

			self.actor_c2_optim.step()
			self.actor_c1_optim.step()

			print("PERFORMANCE:", J)
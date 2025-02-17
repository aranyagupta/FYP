import torch
from torch.distributions import MultivariateNormal



# ----------------- Regular Environments ----------------- #
''' 
    Generates a Witsenhausen environment 
    Environment and system are used interchangeably here
'''
class WitsEnv():
    '''
        Initialises a Witsenhausen Counterexample environment
        k: Witsenhausen cost parameter
        sigma: standard dev of x_0
        dims: dimension of environment (system state variables)
    '''
    def __init__(self, k, sigma, dims, device, mode='TRAIN'):
        torch.set_default_dtype(torch.float64)
        
        self.k = k
        self.sigma = sigma
        self.dims = dims
        self.device = device
        self.mode = mode

        if mode == 'TEST':
            self.x_0 = torch.normal(0, self.sigma, (100000, self.dims), device=self.device)
            self.noise = torch.normal(0, 1, (100000, self.dims), device=self.device)
    
    def step_timesteps(self, actor_c1, actor_c2, timesteps, noise=True):
        if self.mode == 'TEST':
            with torch.no_grad():
                u_1 = actor_c1(self.x_0)
                y_1 = u_1 + noise*self.noise
                u_2 = actor_c2(y_1)
                reward = (self.k**2 * (self.x_0 - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        self.x_0 = torch.normal(0, self.sigma, (timesteps,self.dims), device=self.device)
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1
        
        y_1 = self.x_1 + noise*torch.normal(0, 1, (timesteps,self.dims), device=self.device)
        u_2 = actor_c2(y_1)
        self.x_2 = u_2
        

        terminated = False
        truncated = False

        reward = - (self.k**2 * (self.x_0 - self.x_1)**2 + (self.x_2-self.x_1)**2)

        act_c1 = self.x_1
        act_c2 = self.x_2 
        obs_c1 = self.x_0
        obs_c2 = y_1 
        # obs for c1, obs for c2, act for c1, act for c2, reward, termination or truncation
        return obs_c1, obs_c2, reward, terminated, truncated

class WitsEnvConstrained:
    '''
        Initialises a Witsenhausen Counterexample environment with constrained loss
        k: Witsenhausen cost parameter
        sigma: standard dev of x_0
        dims: dimension of environment (system state variables)
    '''
    def __init__(self, k, sigma, dims, device, mode='TRAIN', constrain_odd = False, constrain_nonaffine = True):
        torch.set_default_dtype(torch.float64)
        
        self.k = k
        self.sigma = sigma
        self.dims = dims
        self.device = device
        self.mode = mode
        self.epsilon = 1
        self.constrain_odd = constrain_odd
        self.constrain_nonaffine = constrain_nonaffine

        if mode == 'TEST':
            self.x_0 = torch.normal(0, self.sigma, (100000, self.dims), device=self.device)
            self.noise = torch.normal(0, 1, (100000, self.dims), device=self.device)
    
    def step_timesteps(self, actor_c1, actor_c2, lamb_0=0, lamb_1=0, mu_0=0, mu_1=0, timesteps=1000000, noise=True):
        if self.mode == 'TEST':
            with torch.no_grad():
                u_1 = actor_c1(self.x_0)
                y_1 = u_1 + noise*self.noise
                u_2 = actor_c2(y_1)
                reward = (self.k**2 * (self.x_0 - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        self.x_0 = torch.normal(0, self.sigma, (timesteps,self.dims), device=self.device)
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1
        
        y_1 = self.x_1 + noise*torch.normal(0, 1, (timesteps,self.dims), device=self.device)
        u_2 = actor_c2(y_1)
        self.x_2 = u_2
        

        terminated = False
        truncated = False

        if self.constrain_nonaffine:
            N = self.x_0.size(0)
            sum_x_1 = torch.sum(self.x_0)
            sum_y_1 = torch.sum(self.x_1)
            sum_xy_1 = torch.sum(self.x_0 * self.x_1)
            sum_x2_1 = torch.sum(self.x_0 * self.x_0)

            first_affine_gradient = (N * sum_xy_1 - sum_x_1 * sum_y_1) / (N * sum_x2_1 - sum_x_1 ** 2)
            first_affine_intercept = (sum_y_1 - first_affine_gradient * sum_x_1) / N

            sum_x_2 = torch.sum(y_1)
            sum_y_2 = torch.sum(self.x_2)
            sum_xy_2 = torch.sum(y_1 * self.x_2)
            sum_x2_2 = torch.sum(self.x_2 * self.x_2)
            
            second_affine_gradient = (N * sum_xy_2 - sum_x_2 * sum_y_2) / (N * sum_x2_2 - sum_x_2 ** 2)
            second_affine_intercept = (sum_y_2 - second_affine_gradient * sum_x_2) / N

        f_x = (self.k**2 * (self.x_0 - self.x_1)**2 + (self.x_2-self.x_1)**2)
        h_0_x = 0
        h_1_x = 0
        g_0_x = 0
        g_1_x = 0

        if self.constrain_odd:
            h_0_x = (actor_c1(-self.x_0) + self.x_1)
            h_1_x = (actor_c2(-y_1) + self.x_2)
        
        if self.constrain_nonaffine:
            g_0_x = (self.epsilon - (self.x_1 - (first_affine_gradient*self.x_0 + first_affine_intercept))**2)
            g_1_x = (self.epsilon - (self.x_2 - (second_affine_gradient*y_1 + second_affine_intercept))**2)
        
        print("f_x:", f_x.mean())
        reward = f_x + lamb_0 * h_0_x + lamb_1 * h_1_x + mu_0 * g_0_x + mu_1 * g_1_x

        obs_c1 = self.x_0
        obs_c2 = y_1 
        # obs for c1, obs for c2, act for c1, act for c2, reward, termination or truncation
        return obs_c1, obs_c2, reward, terminated, truncated

class WitsEnvCombined:
    def __init__(self, k, sigma, dims, device, mode='TRAIN'):
        torch.set_default_dtype(torch.float64)
        
        self.k = k
        self.sigma = sigma
        self.device = device
        self.mode = mode
        self.dims = dims

        if self.mode == 'TEST':
            self.x = torch.normal(0, self.sigma, (100000, self.dims), device=self.device)
            self.noise = torch.normal(0, 1, (100000, self.dims), device=self.device)

    def step_timesteps(self, actor_combined, actor_combined_second, timesteps, noise=True):
        if self.mode=='TEST':
            with torch.no_grad():
                u_1, y_2, u_2 = actor_combined(self.x, noise_data=self.noise)
                reward = (self.k**2 * (x - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        x = torch.normal(0, self.sigma, (timesteps,1), device=self.device)
        
        u_1, y_2, u_2 = actor_combined(x)
        # u_1 = u_1.clone().detach()
        # y_2 = y_2.clone().detach()
        
        obs_c1 = x
        obs_c2 = y_2
        reward = - (self.k**2 * (x - u_1)**2 + (u_2-u_1)**2)
        return obs_c1, obs_c2, reward, False, False

# ----------------- Simplified Environments ----------------- #
class WitsEnvSimple():
    '''
        Initialises a simplified Witsenhausen Counterexample environment
        requiring only the first controller and a fixed, known input
        Used to test if PPO and KAN can actually learn an arbitrary function,
        as the optimal controller here should have 0 loss
        k: Witsenhausen cost parameter
        sigma: standard dev of x_0
        dims: dimension of environment (system state variables)
    '''
    
    def __init__(self, k, sigma, actor_c1, dims, device):
        torch.set_default_dtype(torch.float64)
        
        self.k = k
        self.sigma = sigma
        self.dims = dims
        self.device = device

        self.cov_var = torch.full(size=(self.dims,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)


        self.x_0 = torch.normal(0.0, float(self.sigma), (20,self.dims), device=self.device)
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1

    def f(self, x):
        return torch.sin(x) 

    def reset(self, actor_c1, seed = None):
        super().reset(seed=seed)
        actor_c1.reset()

        self.x_0 = torch.normal(0.0, float(self.sigma), (20,self.dims), device=self.device)
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1

        return self.x_0
    
    
    # runs timesteps number of steps at once
    def step_timesteps(self, actor_c1, timesteps):
        self.x_0 = torch.normal(0.0, float(self.sigma), (timesteps,self.dims), device=self.device)
        # self.x_0 = torch.arange(0, 2*torch.pi, 2*torch.pi/float(timesteps), dtype=torch.float64, device=self.device).reshape(timesteps, self.dims)
        
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1


        terminated = False
        truncated = False
        rewards = -((self.f(self.x_0)-self.x_1)**2)

        observations = self.x_0
        actions = self.x_1
		# Return the sampled action and the log probability of that action in our distribution
        if self.device == torch.device('cuda'):
            return observations, rewards, terminated, truncated,
        else:
            return observations, rewards, terminated, truncated

class WitsActorTestSimple:
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
        self.env = WitsEnvSimple(k=0.2, sigma=5, actor_c1=actor, dims=1, device=self.device)
    
    # find E[-(self.k **2 * (self.f(x_0_obs)-self.x_1)**2)] numerically
    def test(self, timesteps):
        _, rewards, _, _ = self.env.step_timesteps(actor_c1=self.actor, timesteps=timesteps)
        return -rewards.mean()
        
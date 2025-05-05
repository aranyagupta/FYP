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
        torch.set_default_dtype(torch.float32)
        
        self.k = k
        self.sigma = sigma
        self.dims = dims
        self.device = device
        self.mode = mode

        if mode == 'TEST':
            self.x_0 = torch.normal(0, self.sigma, (1000000, self.dims), device=self.device)
            self.noise = torch.normal(0, 1, (1000000, self.dims), device=self.device)
    
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
    def __init__(self, k, sigma, dims, device, mode='TRAIN', constrain_odd = False, constrain_nonlinear=False, constrain_nonaffine = False, constrain_new = True):
        torch.set_default_dtype(torch.float32)
        
        self.k = k
        self.sigma = sigma
        self.dims = dims
        self.device = device
        self.mode = mode
        self.epsilon = 10
        self.constrain_odd = constrain_odd
        self.constrain_nonaffine = constrain_nonaffine
        self.constrain_nonlinear = constrain_nonlinear
        self.constrain_new = constrain_new

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

        if self.constrain_nonlinear:
            first_linear_gradient = (torch.transpose(self.x_0, 0, 1) @ self.x_1)/(torch.transpose(self.x_0, 0, 1) @ self.x_0 + 1e-8) 
            second_linear_gradient = (torch.transpose(y_1, 0, 1) @ self.x_2)/(torch.transpose(y_1, 0, 1) @ y_1 + 1e-8) 

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

        # new constraint approach - require area between f(x) and linear approximation to be nonzero
        # and require f(0) = 0
        if self.constrain_new:
            num_points = 1000
            linear_data_spread = torch.arange(-20.0, 20.0, 40.0/float(num_points))
            linear_data_spread = linear_data_spread.reshape(linear_data_spread.shape[0], 1)

            fun_1 = actor_c1(linear_data_spread)
            fun_2 = actor_c2(linear_data_spread)

            first_linear_gradient = (torch.transpose(linear_data_spread, 0, 1) @ fun_1)/(torch.transpose(linear_data_spread, 0, 1) @ linear_data_spread)
            second_linear_gradient = (torch.transpose(linear_data_spread, 0, 1) @ fun_2)/(torch.transpose(linear_data_spread, 0, 1) @ linear_data_spread)
            
            first_area = torch.trapz(torch.abs(fun_1-first_linear_gradient*linear_data_spread), dx=40.0/float(num_points))
            second_area = torch.trapz(torch.abs(fun_2-second_linear_gradient*linear_data_spread), dx=40.0/float(num_points))

            first_at_zero = actor_c1(torch.zeros(2, 1))[0]
            second_at_zero = actor_c2(torch.zeros(2, 1))[0]


        f_x = (self.k**2 * (self.x_0 - self.x_1)**2 + (self.x_2-self.x_1)**2)
        h_0_x = 0
        h_1_x = 0
        g_0_x = 0
        g_1_x = 0

        if self.constrain_odd:
            h_0_x = (actor_c1(-self.x_0) + self.x_1)
            h_1_x = (actor_c2(-y_1) + self.x_2)
        
        if self.constrain_nonlinear:
            g_0_x = (self.epsilon - (self.x_1 - (first_linear_gradient*self.x_0))**2)
            g_1_x = (self.epsilon - (self.x_2 - (second_linear_gradient*y_1))**2)
        
        if self.constrain_nonaffine:
            g_0_x = (self.epsilon - (self.x_1 - (first_affine_gradient*self.x_0 + first_affine_intercept))**2)
            g_1_x = (self.epsilon - (self.x_2 - (second_affine_gradient*y_1 + second_affine_intercept))**2)

        if self.constrain_new:
            h_0_x = first_at_zero
            h_1_x = second_at_zero

            g_0_x = self.epsilon - first_area
            g_1_x = self.epsilon - second_area
        
        print("f_x:", f_x.mean())
        reward = f_x + lamb_0 * h_0_x + lamb_1 * h_1_x + mu_0 * g_0_x + mu_1 * g_1_x

        obs_c1 = self.x_0
        obs_c2 = y_1 
        # obs for c1, obs for c2, act for c1, act for c2, reward, termination or truncation
        return obs_c1, obs_c2, reward, terminated, truncated

class WitsEnvCombined:
    def __init__(self, k, sigma, dims, device, mode='TRAIN'):
        torch.set_default_dtype(torch.float32)
        
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
    
# Environment for local search algorithm
class WitsEnvLSA:
    def __init__(self, k, sigma, dims, device, mode='TRAIN'):
        torch.set_default_dtype(torch.float32)
        
        self.k = k
        self.sigma = sigma
        self.device = device
        self.mode = mode
        self.dims = dims

        if self.mode == 'TEST':
            TEST_TIMESTEPS = 20000
            self.x_0 = torch.arange(-3*self.sigma, 3*self.sigma, (6*self.sigma)/TEST_TIMESTEPS)
            self.x_0 = self.x_0.reshape(self.x_0.shape[0], 1)
            self.y_1 = torch.arange(-3*self.sigma, 3*self.sigma, (6*self.sigma)/TEST_TIMESTEPS)
            self.y_1 = self.y_1.reshape(self.y_1.shape[0], 1)
    
    # generate u_1 tensor from randomly generated x0
    def generate_u1_tensor_random(self, y1, x1):
        # y1: [N], x1: [N]
        # Goal: For each y1[i], compute weighted sum over x1
        # Expand y1 to [B, N], x1 to [B, N] via broadcasting
        y1_exp = y1.unsqueeze(1)         # [N, 1]
        x1_exp = x1.unsqueeze(0)         # [1, N]
        
        log_weights = -0.5 * (y1_exp - x1_exp) ** 2  # [N, N]
        log_weights = log_weights - torch.max(log_weights, dim=1, keepdim=True).values  # stability
        weights = torch.exp(log_weights)  # [N, N]
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # normalize

        out = torch.sum(weights * x1_exp, dim=1)  # [N]
        return out
    
    # generate u1 tensor from non-randomly generated x0, x1
    def generate_u1_tensor(self, y1, x1, x0):
        f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
        f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))


        x_0_integrating, indices = torch.sort(x0, dim=0)
        sorted_indices = indices.squeeze(1)

        y1_exp = y1.expand(y1.shape[0], y1.shape[0]).T
        bottomIntegrand = f_X(x0)*f_W(y1_exp-x1)
        bottomIntegrand_sorted = bottomIntegrand[sorted_indices, :]
        bottomIntegral = torch.trapz(y=bottomIntegrand_sorted, x=x_0_integrating.squeeze(1), dim=0)

        topIntegrand = x1*bottomIntegrand
        topIntegrand_sorted = topIntegrand[sorted_indices, :]
        topIntegral = torch.trapz(y=topIntegrand_sorted, x=x_0_integrating.squeeze(1), dim=0)

        u1 = topIntegral/bottomIntegral
        return u1



    def step_timesteps(self, actor_c1, actor_c2=None, timesteps=100000):
        # actor_c1 = x1(x0) in paper
        # No need for actor c2 as it can be directly calculated
        if self.mode == 'TEST':
            with torch.no_grad():
                x_1 = actor_c1(self.x_0)
                u_1 = self.generate_u1_tensor(self.y_1, x_1, self.x_0)
                f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
                f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))

                x_0_integrating, x_0_indices = torch.sort(self.x_0, dim=0)
                x_0_sorted_indices = x_0_indices.squeeze(1)

                first_integrand = self.k**2*((x_1-self.x_0)**2 * f_X(self.x_0))
                first_integral = torch.trapz(y=first_integrand[x_0_sorted_indices, :], x=x_0_integrating.squeeze(1), dim=0)

                y_1_integrating, y_1_indices = torch.sort(self.y_1, dim=0)
                y_1_sorted_indices = y_1_indices.squeeze(1)
                x_0_exp = self.x_0.expand(self.x_0.shape[0], self.x_0.shape[0]).T
                x_1_exp = x_1.expand(x_1.shape[0], x_1.shape[0]).T

                second_integrand = (x_1_exp-u_1)**2*f_X(x_0_exp)*f_W(self.y_1-x_0_exp)
                subIntegral = torch.trapz(y=second_integrand[x_0_sorted_indices, :], x=x_0_integrating.squeeze(1), dim=0)
                second_integral = torch.trapz(y=subIntegral[y_1_sorted_indices, :], x=y_1_integrating.squeeze(1), dim=0)
                
                print("first_integral.shape:", first_integral.shape)
                print("second_integral.shape:", second_integral.shape)

                return first_integral + second_integral
            
        # x_0 = torch.normal(0, self.sigma, (timesteps,1), device=self.device)
        # x_1 = actor_c1(x_0)
        # w = torch.normal(0, 1, (timesteps,self.dims), device=self.device)
        # y_1 = x_1 + w
        # x_2 = self.generate_u1_tensor(y_1, x_1)

        x_0 = torch.arange(-3*self.sigma, 3*self.sigma, (6*self.sigma)/timesteps)
        x_0 = x_0.reshape(x_0.shape[0], 1)
        x_1 = actor_c1(x_0)
        y_1 = torch.arange(-3*self.sigma, 3*self.sigma, (6*self.sigma)/timesteps)
        y_1 = y_1.reshape(y_1.shape[0], 1)
        x_2 = self.generate_u1_tensor(y_1, x_1, x_0)

        # u1(y1) fixed, as it can be computed for arbitrary input using generate_u1_tensor
        # now, calculate gradient for x1(x0) using u1(y1)
        
        # x0 distribution pdf
        f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
        # w distribution pdf
        f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))

        # dJ/dx_1[x_1, u_1](x_0) as in paper
        # done by integrating D'_X(x_0, y_1)f_X(x_0)f_W(w) wrt y_1
        # fix x_0, sort y_1 and permute integrand correspondingly, then integrate using trapz
        # store result in dJ/dx_1, then add remaining constant part at the end

        y_1_integrating, indices = torch.sort(y_1, dim=0)
        sorted_indices = indices.squeeze(1)
        dJ_dx1 = 2*self.k**2*(x_1-x_0)*f_X(x_0)     

        x_0_exp = x_0.expand(x_0.shape[0], x_0.shape[0]).T
        x_1_exp = x_1.expand(x_1.shape[0], x_1.shape[0]).T

        integrand = (2*(x_1_exp-x_2) + (y_1-x_1_exp) * (x_1_exp-x_2)**2)*f_X(x_0_exp)*f_W(y_1-x_1_exp)
        integrand_sorted = integrand[sorted_indices, :]
        integral = torch.trapz(y=integrand_sorted, x=y_1_integrating.squeeze(1), dim=0)
        integral = integral.reshape(integral.shape[0], 1)
        dJ_dx1 = dJ_dx1 + integral

        ################ FOR LOOP IMPLEMENTATION - SLOW ###########################
        # dJ_dx1_check = 2*self.k**2*(x_1-x_0)*f_X(x_0)
        # # print("initial dJ_dx1 has nan:", torch.any(torch.isnan(dJ_dx1)))
        # for i in range(x_0.shape[0]):
        #     current_x_0 = x_0[i]
        #     current_x_1 = x_1[i]

        #     # integrand at a fixed (float) value of x_0
        #     integrand = (2*(current_x_1-x_2) + (y_1-current_x_1) * (current_x_1-x_2)**2)*f_X(current_x_0)*f_W(y_1-current_x_1)
        #     # computing integral over all y_1
        #     integral = torch.trapz(y=integrand[indices].reshape(integrand.shape[0], 1), x=y_1_integrating, dim=0)
        #     dJ_dx1_check[i] = dJ_dx1_check[i] + integral
        # print("dJ_dx1:", dJ_dx1)
        # print("dJ_dx1_check:", dJ_dx1_check)
        # assert torch.any(torch.abs(dJ_dx1_check - dJ_dx1)<=1e-4), f"dJ_dx1 is not accurate:\n value={dJ_dx1},\n check={dJ_dx1_check}"
        ############################################################################

        # Partial derivative of dJ_dx1 wrt x_1, taken from paper
        dx1_dJ_dx1 = 2*(self.k**2-1)*f_X(x_0)
        integrand = ((y_1-x_1_exp)*(x_1_exp-x_2+2)**2 - (x_1_exp-x_2))**2 * f_X(x_0_exp)*f_W(y_1-x_1_exp)
        integrand_sorted = integrand[sorted_indices, :]
        integral = torch.trapz(y=integrand_sorted, x=y_1_integrating.squeeze(1), dim=0)
        integral = integral.reshape(integral.shape[0], 1)
        dx1_dJ_dx1 = dx1_dJ_dx1 + integral

         ################ FOR LOOP IMPLEMENTATION - SLOW ###########################
        # dx1_dJ_dx1_check = 2*(self.k**2-1)*f_X(x_0)
        # for i in range(x_0.shape[0]):
        #     current_x_0 = x_0[i]
        #     current_x_1 = x_1[i]
        #     integrand = ((y_1-current_x_1)*(current_x_1-x_2+2)**2 - (current_x_1-x_2))**2 * f_X(current_x_0)*f_W(y_1-current_x_1)
        #     integral = torch.trapz(y=integrand[indices].reshape(integrand.shape[0], 1), x=y_1_integrating, dim=0)
        #     dx1_dJ_dx1_check[i] = dx1_dJ_dx1_check[i] + integral
        # print("dx1_dJ_dx1:", dx1_dJ_dx1)
        # print("dx1_dJ_dx1_check:", dx1_dJ_dx1_check)
        # assert torch.any(torch.abs(dx1_dJ_dx1_check - dx1_dJ_dx1)<=1e-4), f"dx1_dJ_dx1 is not accurate:\n value:{dx1_dJ_dx1},\n check:{dx1_dJ_dx1_check}"
        ############################################################################
        return dJ_dx1, dx1_dJ_dx1, x_1, x_0
   

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
        torch.set_default_dtype(torch.float32)
        
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
        # self.x_0 = torch.arange(0, 2*torch.pi, 2*torch.pi/float(timesteps), dtype=torch.float32, device=self.device).reshape(timesteps, self.dims)
        
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
        
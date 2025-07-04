import torch

'''
Superclass for all Witsenhausen Environments
Every environment should implement this class
'''
class WitsEnvSuper:
    def __init__(self, k, sigma, device, mode='TRAIN'):
        self.k = k
        self.sigma = sigma
        self.device = device
        self.mode = mode

        if mode == 'TEST':
            self.x_0 = torch.normal(0, self.sigma, (1000000, 1), device=self.device)
            self.noise = torch.normal(0, 1, (1000000, 1), device=self.device)

    def step_timesteps(self, actor_c1, actor_c2, timesteps=100000):
        raise Exception("Not Implemented!")

# ----------------- Regular Environments ----------------- #
''' 
Unconstrained Optimisation Environment
'''
class WitsEnv(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN'):
        '''
            Initialises an Unconstrained Optimisation environment
            Args:
                k (float): Witsenhausen cost parameter
                sigma (float): standard dev of x
                device (torch.device): torch GPU/CPU
                mode (string): TRAIN/TEST
        '''
        torch.set_default_dtype(torch.float32)

        super().__init__(k, sigma, device, mode=mode)
    
    def step_timesteps(self, actor_c1, actor_c2, timesteps):
        '''
            Uses MC estimation to produce the Witsenhausen Loss J
            Args:
                actor_c1 (kan.KAN): KAN model representing first controller
                actor_c2 (kan.KAN): KAN model representing second controller
                timesteps (int): number of timesteps to simulate
            Returns:
                reward: Witsenhausen cost J = E[k^2(f(x)-x)^2 + (g(f(x)+eta) - f(x))^2]. 
                Not sure why I called it reward, since Trainer will try to minimise it.
        '''
        if self.mode == 'TEST':
            with torch.no_grad():
                u_1 = actor_c1(self.x_0)
                y_1 = u_1 + self.noise
                u_2 = actor_c2(y_1)
                reward = (self.k**2 * (self.x_0 - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        x_0 = torch.normal(0, self.sigma, (timesteps,1), device=self.device)
        u_1 = actor_c1(x_0)
        x_1 = u_1
        
        y_1 = x_1 + torch.normal(0, 1, (timesteps,1), device=self.device)
        u_2 = actor_c2(y_1)
        x_2 = u_2
        
        reward = (self.k**2 * (x_0 - x_1)**2 + (x_2-x_1)**2)

        return reward

class WitsEnvConstrained(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN', constrain_odd = False, constrain_nonlinear=False, constrain_nonaffine = False, constrain_new = True):
        '''
            Initialises a Witsenhausen Counterexample environment with Lagrangian augmented loss
            Args:
                k (float): Witsenhausen cost parameter
                sigma (float): standard dev of x
                device (torch.device): torch GPU/CPU
                mode (string): TRAIN/TEST
                constrain_odd (bool): use odd equality constraint mu(x) + mu(-x) = 0
                constrain_nonlinear (bool): use nonlinearity inequality constraint epsilon - (mu(x) - x)^2 < 0
                constrain_nonaffine (bool): use nonaffine inequality constraint epsilon - (mu(x) - (mx+c))^2 < 0
                constrain_new (bool): use area constraint (see report for details)
            
        '''
        torch.set_default_dtype(torch.float32)
        super().__init__(k, sigma, device, mode=mode)
        
        self.epsilon = 5
        self.constrain_odd = constrain_odd
        self.constrain_nonaffine = constrain_nonaffine
        self.constrain_nonlinear = constrain_nonlinear
        self.constrain_new = constrain_new
    
    def step_timesteps(self, actor_c1, actor_c2, lamb_0=0, lamb_1=0, mu_0=0, mu_1=0, timesteps=1000000, noise=True):
        '''
            Uses MC estimation to produce the Witsenhausen Loss J
            Args:
                actor_c1 (kan.KAN): KAN model representing first controller
                actor_c2 (kan.KAN): KAN model representing second controller
                lamb_0 (torch.Tensor): equality constraint coefficient for first controller
                lamb_1 (torch.Tensor): equality constraint coefficient for second controller
                mu_0 (torch.Tensor): inequality constraint coefficient for first controller
                mu_1 (torch.Tensor): inequality constraint coefficient for second controller
                timesteps (int): number of timesteps to simulate
                noise (bool): whether or not to use additive noise. Used for testing. 
            Returns:
                reward: Lagrangian cost L = J + lambda * equality constraint + mu * inequality constraint. 
        '''
        if self.mode == 'TEST':
            with torch.no_grad():
                u_1 = actor_c1(self.x_0)
                y_1 = u_1 + noise*self.noise
                u_2 = actor_c2(y_1)
                reward = (self.k**2 * (self.x_0 - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        self.x_0 = torch.normal(0, self.sigma, (timesteps,1), device=self.device)
        u_1 = actor_c1(self.x_0)
        self.x_1 = u_1
        
        y_1 = self.x_1 + noise*torch.normal(0, 1, (timesteps,1), device=self.device)
        u_2 = actor_c2(y_1)
        self.x_2 = u_2

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
            linear_data_spread = torch.linspace(-3*self.sigma, 3*self.sigma, num_points)
            linear_data_spread = linear_data_spread.reshape(linear_data_spread.shape[0], 1)

            fun_1 = actor_c1(linear_data_spread)
            fun_2 = actor_c2(linear_data_spread)

            first_linear_gradient = (torch.transpose(linear_data_spread, 0, 1) @ fun_1)/(torch.transpose(linear_data_spread, 0, 1) @ linear_data_spread)
            second_linear_gradient = (torch.transpose(linear_data_spread, 0, 1) @ fun_2)/(torch.transpose(linear_data_spread, 0, 1) @ linear_data_spread)
            
            first_area = torch.trapz(y=torch.abs(fun_1-first_linear_gradient*linear_data_spread), x=linear_data_spread)
            second_area = torch.trapz(y=torch.abs(fun_2-second_linear_gradient*linear_data_spread), x=linear_data_spread)

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

        return reward

class WitsEnvCombined(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN'):
        torch.set_default_dtype(torch.float32)
        super().__init__(k, sigma, device, mode=mode)
        

    def step_timesteps(self, actor_combined, actor_combined_second=None, timesteps=100000):
        if self.mode=='TEST':
            with torch.no_grad():
                u_1, y_2, u_2 = actor_combined(self.x_0, noise_data=self.noise)
                reward = (self.k**2 * (x - u_1)**2 + (u_2-u_1)**2)
                return reward.mean()
        
        x = torch.normal(0, self.sigma, (timesteps,1), device=self.device)
        
        u_1, y_2, u_2 = actor_combined(x)
        # u_1 = u_1.clone().detach()
        # y_2 = y_2.clone().detach()
        
        reward = (self.k**2 * (x - u_1)**2 + (u_2-u_1)**2)
        return reward
    
# Environment for local search algorithm
class WitsEnvLSA(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN'):
        '''
            Initialises a Local Search Algorithm environment
            Args:
                k (float): Witsenhausen cost parameter
                sigma (float): standard dev of x
                device (torch.device): torch GPU/CPU
                mode (string): TRAIN/TEST            
        '''
        torch.set_default_dtype(torch.float32)
        super().__init__(k, sigma, device, mode=mode)
        
        if self.mode == 'TEST':
            TEST_TIMESTEPS = 20000
            self.x_0 = torch.normal(0, self.sigma, (TEST_TIMESTEPS,1), device=self.device)
            self.noise = torch.normal(0, 1, (TEST_TIMESTEPS, 1), device=self.device)
            
    # generate u_1 tensor from randomly generated x0, given set of y2 values
    def generate_u1_tensor_random(self, y2, x1):
        '''
            Given x1 = f(x) + eta (where x and eta sampled randomly), calculates g(y2)
            Args:
                y2 (torch.tensor): values of y2 at which we want to calculate g(y2) 
                x1 (torch.tensor): values of f(x)+eta generated by MC estimation and first controller
            Returns:
            out (torch.tensor): g(y2), created through importance sampling
        '''
        # uses importance sampling 
        # the probability of finding y given a fixed x1 value is proportional to e^(-(y-x1)^2/2)
        # (as w is distributed normally with mean 0 variance 1)
        # we treat this probability as an importance weighting for that given value of y
        # we then find the expected value across all values of x1 ie 1/N * sum across all x1 of e^(-(y-x1)^2/2)*x1
        # to find the expected value of x1 | y = x1 + w
        y2_exp = y2.unsqueeze(1)
        x1_exp = x1.unsqueeze(0)
        log_weights = -0.5 * (y2_exp - x1_exp) ** 2  
        log_weights = log_weights - torch.max(log_weights, dim=1, keepdim=True).values  #stability
        weights = torch.exp(log_weights)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # normalize

        out = torch.sum(weights * x1_exp, dim=1)  # [N]
        
        return out
    
    # generate u1 tensor from non-randomly generated x0, x1, for a given set of y2 values
    # this explicity calculates the required integrals then uses them to generate y2
    # slower but less prone to noise than the random method
    def generate_u1_tensor(self, y2, x1, x0):
        '''
            Given x1 = f(x) + eta, and x0 = x, where x is evenly sampled from some range, calculates g(y2)
            Args:
                y2 (torch.tensor): values of y2 at which we want to calculate g(y2) 
                x1 (torch.tensor): values of f(x)+eta generated by first controller
                x0 (torch.tensor): values of x evenly sampled from some range
            Returns:
            out (torch.tensor): g(y2), created through integration
        '''
        f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
        f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))

        x_0_integrating, indices = torch.sort(x0, dim=0)
        sorted_indices = indices.squeeze(1)

        y2_exp = y2.expand(y2.shape[0], y2.shape[0]).T
        bottomIntegrand = f_X(x0)*f_W(y2_exp-x1)
        bottomIntegrand_sorted = bottomIntegrand[sorted_indices, :]
        bottomIntegral = torch.trapz(y=bottomIntegrand_sorted, x=x_0_integrating.squeeze(1), dim=0)

        topIntegrand = x1*bottomIntegrand
        topIntegrand_sorted = topIntegrand[sorted_indices, :]
        topIntegral = torch.trapz(y=topIntegrand_sorted, x=x_0_integrating.squeeze(1), dim=0)

        u1 = topIntegral/bottomIntegral
        return u1

    def step_timesteps(self, actor_c1, actor_c2=None, timesteps=100000):
        '''
            Calculates gradients for output tensors
            Args:
                actor_c1 (kan.KAN): model representing first controller 
                timesteps (int): number of divisions
            Returns:
                dJ_dx1 (torch.Tensor): first derivative of loss at x1
                dx1_dJ_dx1 (torch.Tensor): second derivative of loss at x1
                x1 (torch.Tensor): output of first controller
                x0 (torch.Tensor): input to first controller (for stop condition calculation)
        '''
        # actor_c1 = x1(x0) in paper
        # No need for actor c2 as it can be directly calculated
        if self.mode == 'TEST':
            with torch.no_grad():
                x_1 = actor_c1(self.x_0)
                y_2 = x_1 + self.noise
                u_1 = self.generate_u1_tensor_random(y_2, x_1)
                reward = self.k**2 * (self.x_0 - x_1)**2 + (u_1 - x_1)**2
                return reward.mean()
            
        x_0 = torch.linspace(-3*self.sigma, 3*self.sigma, timesteps)
        x_0 = x_0.reshape(x_0.shape[0], 1)
        x_1 = actor_c1(x_0)
        y_2 = torch.linspace(-3*self.sigma, 3*self.sigma, timesteps)
        y_2 = y_2.reshape(y_2.shape[0], 1)
        x_2 = self.generate_u1_tensor(y_2, x_1, x_0)
        # u1(y_2) fixed, as it can be computed for arbitrary input using generate_u1_tensor
        # now, calculate gradient for x1(x0) using u1(y_2)
        
        # x0 distribution pdf
        f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
        # w distribution pdf
        f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))

        # dJ/dx_1[x_1, u_1](x_0) as in paper
        # done by integrating D'_X(x_0, y_2)f_X(x_0)f_W(w) wrt y_2
        # fix x_0, sort y_2 and permute integrand correspondingly, then integrate using trapz
        # store result in dJ/dx_1, then add remaining constant part at the end

        y_2_integrating, indices = torch.sort(y_2, dim=0)
        sorted_indices = indices.squeeze(1)
        dJ_dx1 = 2*self.k**2*(x_1-x_0)*f_X(x_0)     

        x_0_exp = x_0.expand(x_0.shape[0], x_0.shape[0]).T
        x_1_exp = x_1.expand(x_1.shape[0], x_1.shape[0]).T

        integrand = (2*(x_1_exp-x_2) + (y_2-x_1_exp) * (x_1_exp-x_2)**2)*f_X(x_0_exp)*f_W(y_2-x_1_exp)
        integrand_sorted = integrand[sorted_indices, :]
        integral = torch.trapz(y=integrand_sorted, x=y_2_integrating.squeeze(1), dim=0)
        integral = integral.reshape(integral.shape[0], 1)
        dJ_dx1 = dJ_dx1 + integral

        # Partial derivative of dJ_dx1 wrt x_1, taken from paper
        dx1_dJ_dx1 = 2*(self.k**2-1)*f_X(x_0)
        integrand = ((y_2-x_1_exp)*(x_1_exp-x_2+2)**2 - (x_1_exp-x_2))**2 * f_X(x_0_exp)*f_W(y_2-x_1_exp)
        integrand_sorted = integrand[sorted_indices, :]
        integral = torch.trapz(y=integrand_sorted, x=y_2_integrating.squeeze(1), dim=0)
        integral = integral.reshape(integral.shape[0], 1)
        dx1_dJ_dx1 = dx1_dJ_dx1 + integral

        return dJ_dx1, dx1_dJ_dx1, x_1, x_0
   

class WitsEnvFGD(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN'):
        '''
            Initialises Frechet Gradient Descent environment
            Args:
                k (float): Witsenhausen cost parameter
                sigma (float): standard dev of x
                device (torch.device): torch GPU/CPU
                mode (string): TRAIN/TEST
        '''
        super().__init__(k, sigma, device, mode=mode)
        if mode == 'TEST':
            TEST_TIMESTEPS = 20000
            self.x_0 = torch.normal(0, self.sigma, (TEST_TIMESTEPS,1), device=self.device)
            self.noise = torch.normal(0, 1, (TEST_TIMESTEPS, 1), device=self.device)

    def step_timesteps(self, actor_c1, actor_c2, timesteps=10000):
        '''
            Calculates (MC estimated) gradients for output tensors 
            Args:
                actor_c1 (kan.KAN): model representing first controller
                actor_c2 (kan.KAN): model representing second controller 
                timesteps (int): number of timesteps
            Returns:
                frechet_grad_1 (torch.Tensor): Riesz representation of Frechet Gradient of loss for first controller 
                frechet_grad_2 (torch.Tensor): Riesz representation of Frechet Gradient of loss for second controller 
                x1 (torch.Tensor): output of first controller
                x2 (torch.Tensor): output of second controller
                J (torch.Tensor): MC estimated Witsenhausen loss
        '''
        if self.mode == 'TEST':
            with torch.no_grad():
                x1 = actor_c1(self.x_0)
                y1 = self.noise + x1
                x2 = actor_c2(y1)

                reward = self.k**2 * (x1 - self.x_0) ** 2 + (x2-x1)**2
                return reward.mean()

        # wrap to allow jacobian vector product to work
        def mu_1(x):
            return actor_c1(x)
        
        def mu_2(x):
            return actor_c2(x)


        x0 = torch.normal(0, self.sigma, (timesteps, 1), device=self.device)

        ones = torch.ones_like(x0, device=self.device)
        # calculates mu_1(x0) and dmu_1(y)/dy at y = x0
        x1, dmu_1_dy = torch.func.jvp(mu_1, (x0,), (ones,))
        if torch.any(torch.isnan(x1)):
            print("x1 has nan")
        if torch.any(torch.isinf(x1)):
            print("x1 has inf")

        if torch.any(torch.isnan(dmu_1_dy)):
            print("dmu_1_dy has nan")
        if torch.any(torch.isinf(dmu_1_dy)):
            print("dmu_1_dy has inf")

        noise = torch.normal(0, 1, (timesteps, 1), device=self.device)
        y2 = x1 + noise


        # calculates mu_2(y2) and dmu_2(y)/dy at y = mu_1(x0)+eta
        x2, dmu_2_dy = torch.func.jvp(mu_2, (y2,), (ones,))
        # calculates dmu_2(y)/dy at y=x0
        _, dmu_2_dx = torch.func.jvp(mu_2, (x0,), (ones,))

        if torch.any(torch.isnan(x2)):
            print("x2 has nan")
        if torch.any(torch.isinf(x2)):
            print("x2 has inf")

        if torch.any(torch.isnan(dmu_2_dy)):
            print("dmu_2_dy has nan")
        if torch.any(torch.isinf(dmu_2_dy)):
            print("dmu_2_dy has inf")

        if torch.any(torch.isnan(dmu_2_dx)):
            print("dmu_2_dx has nan")
        if torch.any(torch.isinf(dmu_2_dx)):
            print("dmu_2_dx has inf")
        
        frechet_grad_1 = 2*self.k**2*(x1-x0) + 2*(x1-x2)*(1-dmu_2_dy)
        frechet_grad_2 = -2*(x1 - x2)

        if torch.any(torch.isnan(frechet_grad_2)):
            print("frechet_grad_2 has nan")
        if torch.any(torch.isinf(frechet_grad_2)):
            print("frechet_grad_2 has inf")

        J = 0
        with torch.no_grad():
            J = self.k**2*(x1-x0)**2 + (x2-x1)**2
            J = J.mean()

        return frechet_grad_1, frechet_grad_2, x1, x2, J 

# momentum-based frechet gradient descent
# implements zeta_k+1 = beta * zeta_k + (1-beta)*frechetgradient_k
# mu_k+1 = mu_k - tau*zeta_k+1
class WitsEnvMomentum(WitsEnvSuper):
    def __init__(self, k, sigma, device, mode='TRAIN', beta=0.05):
        '''
            Initialises Polyak Momentum + Frechet Gradient Descent environment
            Args:
                k (float): Witsenhausen cost parameter
                sigma (float): standard dev of x
                device (torch.device): torch GPU/CPU
                mode (string): TRAIN/TEST
                beta (float): momentum hyperparameter dictating how "heavy" Polyak's heavy ball is
        '''
        super().__init__(k, sigma, device, mode=mode)
        self.beta = beta
        TEST_TIMESTEPS = 100000
        self.x_0 = torch.normal(0, self.sigma, (TEST_TIMESTEPS,1), device=self.device)
        self.noise = torch.normal(0, 1, (TEST_TIMESTEPS, 1), device=self.device)
        self.zeta_1 = torch.zeros((1,1))
        self.zeta_2 = torch.zeros((1,1))
    
    def step_timesteps(self, actor_c1, actor_c2, timesteps=2000):
        '''
            Calculates descent direction for Polyak's heavy ball method
            Args:
                actor_c1 (kan.KAN): model representing first controller
                actor_c2 (kan.KAN): model representing second controller 
                timesteps (int): number of timesteps
            Returns:
                zeta_1 (torch.Tensor): Updated Polyak descent direction for first controller
                zeta_2 (torch.Tensor): Updated Polyak descent direction for first controller
                x1 (torch.Tensor): output of first controller
                x2 (torch.Tensor): output of second controller
                J (torch.Tensor): MC estimated Witsenhausen loss
        '''
        if self.mode == 'TEST':
            with torch.no_grad():
                x1 = actor_c1(self.x_0)
                y1 = self.noise + x1
                x2 = actor_c2(y1)

                reward = self.k**2 * (x1 - self.x_0) ** 2 + (x2-x1)**2
                return reward.mean()
        
        if self.zeta_1.shape == torch.Size([1, 1]):
            self.zeta_1.expand((timesteps, 1))
            self.zeta_2.expand((timesteps, 1))
        elif self.zeta_1.shape != torch.Size([timesteps, 1]):
            raise Exception("Changed timesteps in between training - not allowed!")
        
        f_X = lambda x : 1.0/(self.sigma* torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-x**2/(2*self.sigma**2))
        f_W = lambda w : 1.0/(torch.sqrt(2*torch.tensor(torch.pi, device=self.device))) * torch.exp(-w**2/(2))
        # wrap to allow jacobian vector product to work        
        def mu_2(x):
            return actor_c2(x)


        x0 = torch.linspace(-3*self.sigma, 3*self.sigma, steps=timesteps, device=self.device)
        x0 = x0.reshape((timesteps, 1))

        # calculates mu_1(x0)
        x1 = actor_c1(x0)


        y2_int = torch.linspace(-3*self.sigma, 3*self.sigma, steps=timesteps, device=self.device)
        y2 = y2_int.reshape((timesteps, 1))
        y2_exp = y2.expand((timesteps, timesteps)).T
        ones = torch.ones_like(y2, device=self.device)

        # calculates mu_2(y2) and dmu_2(y)/dy at y = mu_1(x0)+eta
        x2, dmu_2_dy = torch.func.jvp(mu_2, (y2,), (ones,))
        x2_exp = x2.expand(timesteps, timesteps).T
        dmu_2_dy_exp = dmu_2_dy.expand(timesteps, timesteps).T

        if torch.any(torch.isnan(x2)):
            print("x2 has nan")
        if torch.any(torch.isinf(x2)):
            print("x2 has inf")

        if torch.any(torch.isnan(dmu_2_dy)):
            print("dmu_2_dy has nan")
        if torch.any(torch.isinf(dmu_2_dy)):
            print("dmu_2_dy has inf")

        
        frechet_grad_1 = 2*self.k**2*(x1-x0)*f_X(x0)
        integrand_1 = 2*f_X(x0)*(x2_exp-x1)*(dmu_2_dy_exp-1)*f_W(y2_exp-x1)
        integral_1 = torch.trapz(integrand_1, y2_int, dim=1).reshape(timesteps, 1)
        frechet_grad_1 = frechet_grad_1 + integral_1
        integrand_2 = 2*(x2_exp-x1)*f_X(x0)*f_W(y2_exp-x1)
        frechet_grad_2 = torch.trapz(integrand_2, x0, dim=0).reshape(timesteps, 1)

        self.zeta_1 = self.beta*self.zeta_1.detach() + (1-self.beta)*frechet_grad_1.detach()
        self.zeta_2 = self.beta*self.zeta_2.detach() + (1-self.beta)*frechet_grad_2.detach()

        J = 0
        with torch.no_grad():
            J1 = self.k**2*(x1-x0)**2*f_X(x0)
            J2 = (x2_exp-x1)**2*f_X(x0)*f_W(y2_exp-x1)

            J1 = torch.trapz(J1, x0, dim=0)
            J2 = torch.trapz(J2, y2_int, dim=1)
            J2 = J2.reshape(timesteps, 1)
            J2 = torch.trapz(J2, x0, dim=0)

            J = J1 + J2

        return self.zeta_1, self.zeta_2, x1, x2, J 
    
# ----------------- Simplified Environments ----------------- #
class WitsEnvSimple():
    '''
        Initialises a simplified Witsenhausen Counterexample environment
        requiring only the first controller and a fixed, known input
        Used to test if KAN can actually learn an arbitrary function,
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
        
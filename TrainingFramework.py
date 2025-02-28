import WitsEnv
import WitsPPO
import CombinedKan
import kan
import torch
import os

class TrainingFramework:
    def __init__(self, KAN_hyps=[[1,2,2,1]], k_range=[0.05, 1.0, 0.05], sigma_range=[0.1, 7.1, 0.25], noiseless = False, store_models=True, store_loss=True):
        if len(k_range)==3:
            self.k_range = [k_range[0]+i*k_range[2] for i in range(int( (k_range[1]-k_range[0]) / k_range[2])+1)]
        else:
            self.k_range = k_range
        
        if len(sigma_range)==3:
            self.sigma_range = [sigma_range[0]+i*sigma_range[2] for i in range(int( (sigma_range[1]-sigma_range[0]) / sigma_range[2])+1)]
        else:
            self.sigma_range = sigma_range
        
        self.KAN_hyps = KAN_hyps
        self.store_models = store_models
        self.store_loss = store_loss
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _store_actors(self, model_type, actor_c1, actor_c2, k, sigma, kan_hyp):
        if self.store_models:
            actor_c1.saveckpt(f"./wits_models/{model_type}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}-c1")
            actor_c2.saveckpt(f"./wits_models/{model_type}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}-c2")

    def _store_loss(self, model_type, cost, k, sigma, kan_hyp):
        if self.store_loss:
            with open(f'./wits_models_loss/{model_type}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}.log', 'a') as file:
                file.write(f"test cost: {cost.item()}\n")
    
    # check if a model exists already in wits_models
    # if it does, return true
    def _check_exists(self, model_type, k, sigma, hyp):
        files = os.listdir('wits_models')
        query = f"{model_type}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{hyp}-c1_state"
        for file in files:
            if query==file:
                return True
        return False
    
    # prefit model to initial function 
    def _prefit_to(self, model, function, grid_range=[-20.0, 20.0]):
        assert isinstance(model, kan.KAN), "Model is not default kan, cannot prefit"
        dataset = {
            'train_input':[],
            'train_label':[],
            'test_input':[],
            'test_label':[],
        }
        dataset['train_input'] = torch.arange(grid_range[0], grid_range[1], (grid_range[1]-grid_range[0])/100000.0).to(self.device)
        dataset['train_input'] = dataset['train_input'].reshape(dataset['train_input'].shape[0], 1)
        dataset['test_input'] = torch.arange(grid_range[0], grid_range[1], (grid_range[1]-grid_range[0])/100000.0).to(self.device)
        dataset['test_input'] = dataset['test_input'].reshape(dataset['test_input'].shape[0], 1)
        # assume function requires sigma as argument
        if function.__code__.co_argcount == 2:
            dataset['train_label'] = function(grid_range[1]/3.0, dataset['train_input'])
            dataset['test_label'] = function(grid_range[1]/3.0, dataset['test_input'])
        else:
            dataset['train_label'] = function(dataset['train_input'])
            dataset['test_label'] = function(dataset['test_input'])
        model.fit(dataset, opt="LBFGS", steps=200, lamb=0.)


    def train_framework(self, kanType, env, gradDesc, modelType, prefit_func_1=None, prefit_func_2=None):
        for kan_hyp in self.KAN_hyps:
            for sigma in self.sigma_range:
                grid_range = [-3*sigma, 3*sigma]
                grid = min(int(3*sigma+1), 11)
                prefit_model_1 = kanType(width=kan_hyp, grid=grid, k=3, seed=42, grid_range=grid_range, device=self.device)
                prefit_model_2 = kanType(width=kan_hyp, grid=grid, k=3, seed=42, grid_range=grid_range, device=self.device)
                finished_sigma = [self._check_exists(modelType, k, sigma, kan_hyp) for k in self.k_range]
                finished_sigma = all(finished_sigma)
                if prefit_func_1 is not None and not finished_sigma:
                    self._prefit_to(prefit_model_1, prefit_func_1, grid_range=grid_range)
                if prefit_func_2 is not None and not finished_sigma:
                    self._prefit_to(prefit_model_2, prefit_func_2, grid_range=grid_range)
                for k in self.k_range:
                    testEnv = env(k, sigma, dims=1, device=self.device, mode='TEST')
                    print(f"TRAINING: {modelType}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}")
                    actor_c1 = kanType(width=kan_hyp, grid=grid, k=3, seed=42, grid_range=grid_range, device=self.device)
                    actor_c2 = kanType(width=kan_hyp, grid=grid, k=3, seed=42, grid_range=grid_range, device=self.device)
                    if self._check_exists(modelType, k, sigma, kan_hyp):
                        print(f"SKIPPING: {modelType}-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}, already exists")
                        continue
                    if prefit_func_1 is not None:
                        actor_c1 = prefit_model_1.copy()
                    if prefit_func_2 is not None:
                        actor_c2 = prefit_model_2.copy()
                    trainEnv = env(k, sigma, dims=1, mode='TRAIN', device=self.device)
                    alg = gradDesc(trainEnv, actor_c1, actor_c2)
                    
                    best_loss = 1e7
                    alg.train(100000, 1000)
                    
                    loss = testEnv.step_timesteps(actor_c1, actor_c2, timesteps=100000)
                    print("TEST LOSS:", loss)

                    while (loss < best_loss):
                        self._store_actors(modelType, actor_c1, actor_c2, k, sigma, kan_hyp)
                        self._store_loss(modelType, loss, k, sigma, kan_hyp)
                        
                        best_loss = loss
                        alg.train(100000, 100)
                        loss = testEnv.step_timesteps(actor_c1, actor_c2, timesteps=100000)
                        print("TEST LOSS:", loss)



    def train_dgd(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                testEnv = WitsEnv.WitsEnv(k, sigma, dims=1, device=self.device, mode='TEST')
                for kan_hyp in self.KAN_hyps:
                    print(f"TRAINING: DGD-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}")
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor_c1 = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)).item(), grid_range=grid_range, device=self.device)
                    actor_c2 = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)).item(), grid_range=grid_range, device=self.device)
                    
                    # skip training model if it already exists
                    if self._check_exists('DGD', k, sigma, kan_hyp):
                        print(f"SKIPPING: DGD-k-{k:.2f}-sigma-{sigma:.2f}-hyps-{kan_hyp}, already exists")
                        continue
                        
                    env = WitsEnv.WitsEnv(k=k, sigma=sigma, dims=1, device=self.device)

                    gradDesc = WitsPPO.WitsGradDesc(env, actor_c1, actor_c2, noise=True)
                    
                    best_loss = 1e7
                    gradDesc.train(1000, 1000)
                    
                    loss = testEnv.step_timesteps(actor_c1, actor_c2, 100000)
                    print("TEST LOSS:", loss)

                    while (loss < best_loss):
                        self._store_actors("DGD", actor_c1, actor_c2, k, sigma, kan_hyp)
                        self._store_loss("DGD", loss, k, sigma, kan_hyp)
                        
                        best_loss = loss
                        gradDesc.train(1000, 100)
                        loss = testEnv.step_timesteps(actor_c1, actor_c2, 100000)

    def train_dgd_combined(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                testEnv = WitsEnv.WitsEnvCombined(actor, env, self.device, mode='TEST')
                for kan_hyp in self.KAN_hyps:
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor = CombinedKan.CombinedKan(kan_hyp, grid, 3, torch.randint(low=0, high=2025, size=(1,1)), grid_range, self.device, noise=True)
                    env = WitsEnv.WitsEnvCombined(k, sigma, actor, self.device)

                    gradDesc = WitsPPO.WitsGradDescCombined(env, actor)
                    
                    best_loss = 1e7
                    gradDesc.train(4000, 250)
                    
                    loss = testEnv.step_timesteps(100000, actor)
                    while (loss < best_loss):
                        self._store_actors("DGDCOMB",actor.actor_c1, actor.actor_c2, k, sigma, kan_hyp)
                        self._store_loss("DGDCOMB", loss, k, sigma, kan_hyp)
                                                
                        best_loss = loss
                        gradDesc.train(1000, 100)
                        loss = testEnv.step_timesteps(100000)


    def train_ppo(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                testEnv = WitsEnv.WitsEnvCombined(k, sigma, self.device)
                for kan_hyp in self.KAN_hyps:
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor = CombinedKan.CombinedKan(kan_hyp, grid, 3, torch.randint(low=0, high=2025, size=(1,1)), grid_range, self.device)
                    critic = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)), grid_range=grid_range, device=self.device)
                    env = WitsEnv.WitsEnvCombined(k, sigma, self.device)

                    ppo = WitsPPO.WitsPPOCombined(actor, critic, env)
                    
                    best_loss = 1e7
                    ppo.learn(30000)
                    loss = testEnv.step_timesteps(100000, actor)
                    # test = WitsEnv.WitsActorTestCombined(actor, env, self.device)
                    # loss = test.test(100000)
                    while (loss < best_loss):
                        self._store_actors("PPO",actor.actor_c1, actor.actor_c2, k, sigma, kan_hyp)
                        self._store_loss("PPO", loss, k, sigma, kan_hyp)
                                                
                        best_loss = loss
                        ppo.learn(10000)
                        loss = testEnv.step_timesteps(100000)
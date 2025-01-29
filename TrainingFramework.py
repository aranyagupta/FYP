import WitsEnv
import WitsPPO
import CombinedKan
import kan
import torch

class TrainingFramework:
    def __init__(self, KAN_hyps=[[1,2,2,1]], k_range=[0.05, 1.0, 0.05], sigma_range=[0.1, 7.1, 0.25], noiseless = False, store_models=True, store_loss=True):
        self.k_range = [k_range[0]+i*k_range[2] for i in range(int( (k_range[1]-k_range[0]) / k_range[2])+1)]
        self.sigma_range = [sigma_range[0]+i*sigma_range[2] for i in range(int( (sigma_range[1]-sigma_range[0]) / sigma_range[2])+1)]
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
    

    def train_dgd(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                for kan_hyp in self.KAN_hyps:
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor_c1 = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)).item(), grid_range=grid_range, device=self.device)
                    actor_c2 = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)).item(), grid_range=grid_range, device=self.device)
                    env = WitsEnv.WitsEnv(k=k, sigma=sigma, actor_c1=actor_c1, actor_c2=actor_c2, dims=1, device=self.device)

                    gradDesc = WitsPPO.WitsGradDesc(env, actor_c1, actor_c2, noise=True)
                    
                    best_loss = 1e7
                    gradDesc.train(1000, 100)
                    
                    testEnv = WitsEnv.WitsEnv(k, sigma, actor_c1, actor_c2, dims=1, device=self.device, mode='TEST')
                    loss = testEnv.step_timesteps(actor_c1, actor_c2, 100000)

                    while (loss < best_loss):
                        self._store_actors("DGD", actor_c1, actor_c2, k, sigma, kan_hyp)
                        self._store_loss("DGD", loss, k, sigma, kan_hyp)
                        
                        best_loss = loss
                        gradDesc.train(1000, 100)
                        loss = testEnv.step_timesteps(actor_c1, actor_c2, 100000)

    def train_dgd_combined(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                for kan_hyp in self.KAN_hyps:
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor = CombinedKan.CombinedKan(kan_hyp, grid, 3, grid_range, self.device, noise=True)
                    env = WitsEnv.WitsEnvCombined(k, sigma, actor, self.device)

                    gradDesc = WitsPPO.WitsGradDescCombined(env, actor)
                    
                    best_loss = 1e7
                    gradDesc.train(3000, 100)
                    
                    testEnv = WitsEnv.WitsEnvCombined(actor, env, self.device, mode='TEST')
                    loss = testEnv.step_timesteps(100000)
                    while (loss < best_loss):
                        self._store_actors("DGDCOMB",actor.actor_c1, actor.actor_c2, k, sigma, kan_hyp)
                        self._store_loss("DGDCOMB", loss, k, sigma, kan_hyp)
                                                
                        best_loss = loss
                        gradDesc.train(1000, 100)
                        loss = testEnv.step_timesteps(100000)


    def train_ppo(self):
        for k in self.k_range:
            for sigma in self.sigma_range:
                for kan_hyp in self.KAN_hyps:
                    grid_range = [-3*sigma, 3*sigma]
                    grid = min(int(3*sigma+1), 11)
                    actor = CombinedKan.CombinedKan(kan_hyp, grid, 3, grid_range, self.device, noise=True)
                    critic = kan.KAN(width=kan_hyp, grid=grid, k=3, seed=torch.randint(low=0, high=2025, size=(1,1)), grid_range=grid_range, device=self.device)
                    env = WitsEnv.WitsEnvCombined(k, sigma, actor, self.device)

                    ppo = WitsPPO.WitsPPOCombined(actor, critic, env)
                    
                    best_loss = 1e7
                    ppo.learn(30000)
                    testEnv = WitsEnv.WitsEnvCombined(k, sigma, actor, self.device)
                    loss = testEnv.step_timesteps(100000)
                    # test = WitsEnv.WitsActorTestCombined(actor, env, self.device)
                    # loss = test.test(100000)
                    while (loss < best_loss):
                        self._store_actors("PPO",actor.actor_c1, actor.actor_c2, k, sigma, kan_hyp)
                        self._store_loss("PPO", loss, k, sigma, kan_hyp)
                                                
                        best_loss = loss
                        ppo.learn(10000)
                        loss = testEnv.step_timesteps(100000)
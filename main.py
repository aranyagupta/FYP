import torch
import TrainingFramework
from DataVis import *
from SplineComposer import *
import kan
import CombinedKan
import WitsEnv
import WitsPPO

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_device(device=device)

TRAIN_DGD = False
TRAIN_PPO = False
TRAIN_DGDCOMB = False
TRAIN_ALTERNATING = False
TRAIN_LAG = True

DISPLAY_HEATMAP = False
DISPLAY_SYMBOLIC = False

PLOT_GRAPHS = False

if __name__ == "__main__":
    min_hidden_layers = 3
    max_hidden_layers = 3
    min_layer_width = 2
    max_layer_width = 6

    kvals = torch.sqrt(torch.arange(0.05, 0.35, 0.05)).tolist()
    sigvals = torch.sqrt(torch.arange(5.0, 45.0, 5.0)).tolist()
    kan_hyps = []
    for num_layers in range(min_hidden_layers, max_hidden_layers+1):
        for layer_width in range(min_layer_width, max_layer_width+1):
            hidden = [layer_width] * num_layers
            hyps = [1] + hidden + [1]
            kan_hyps.append(hyps)

    f = TrainingFramework.TrainingFramework(k_range=kvals, sigma_range=sigvals, KAN_hyps=kan_hyps)
    if TRAIN_DGD:
        kanType = kan.KAN
        env = WitsEnv.WitsEnv
        gradDesc = WitsPPO.WitsGradDesc
        modelType = 'DGD'
        f.train_framework(kanType, env, gradDesc, modelType, prefit_func_1=lambda sigma, x : sigma * torch.sign(x), prefit_func_2=lambda sigma, x: sigma*torch.tanh(sigma*x))
    if TRAIN_PPO:
        kanType = CombinedKan.CombinedKan
        env = WitsEnv.WitsEnvCombined
        gradDesc = WitsPPO.WitsPPOCombined
        modelType = 'PPO'
        f.train_framework(kanType, env, gradDesc, modelType)
    if TRAIN_DGDCOMB:
        kanType = CombinedKan.CombinedKan
        env = WitsEnv.WitsEnvCombined
        gradDesc = WitsPPO.WitsGradDescCombined
        modelType = 'DGDCOMB'
        f.train_framework(kanType, env, gradDesc, modelType)
    if TRAIN_ALTERNATING:
        kanType = kan.KAN
        env = WitsEnv.WitsEnv
        gradDesc = WitsPPO.WitsAlternatingDescent
        modelType = "ALTERNATING"
        f.train_framework(kanType, env, gradDesc, modelType)
    if TRAIN_LAG:
        kanType = kan.KAN
        env = WitsEnv.WitsEnvConstrained
        gradDesc = WitsPPO.WitsGradDescConstrained
        modelType = "LAG"
        f.train_framework(kanType, env, gradDesc, modelType)

    if DISPLAY_HEATMAP:
        hyps =  [[1,0],[2,0],[2,0],[2,0],[1,0]]
        kvals, sigvals, losses = getLosses(dgd=False, dgdcomb=False, ppo=False, lag=True, hyps=hyps, models_loss_dir="./constrained_experiments/lag_constrained_area_and_origin_loss/")

        kvals_squared = [round(k**2/0.05)*0.05 for k in kvals]
        varvals = [round(s**2/5.0)*5.0 for s in sigvals]
        losses = [min(2.0, x) for x in losses]

        # lookup = generateLookupTable(kvals, sigvals, losses)
        # print(lookup[(0.3, 4.6)])

        modelType = 'LAG'

        create_heatmap(kvals_squared, varvals, losses, cmap='plasma', title=f"{modelType} {[x[0] for x in hyps]} Model Costs (Area constraint)")
    
    if DISPLAY_SYMBOLIC:
        hyps = [[1,0],[2,0],[2,0],[2,0],[1,0]]

        # LINEAR (UNINTENTIONAL - SHOULD BE 3-STEP)
        k = 0.22
        sigma = "2.24"  

        # 3 STEP: TBF
        # k = 0.22
        # sigma = "5.00"

        # 5-STEP (INTENTIONAL)
        # k = 0.39
        # sigma = 3.87

        modelType = 'LAG'
        
        name = f"constrained_experiments/lag_constrained_area_and_origin_models/{modelType}-k-{k}-sigma-{sigma}-hyps-{hyps}-"
        actor_c1 = kan.KAN.loadckpt(name+"c1")
        actor_c2 = kan.KAN.loadckpt(name+"c2")
        act_fun_c1 = actor_c1.act_fun
        act_fun_c2 = actor_c2.act_fun
        
        if PLOT_GRAPHS:
            # actor_c1.plot()
            # plt.show()
            # actor_c2.plot()
            # plt.show()
        
            plot_model_bruteforce(actor_c1, device=device, range=(-20.0, 20.0), title=f"Reconstruction: C1, k={k}, sig={sigma}, {modelType} Wits Preinit")
            plot_model_bruteforce(actor_c2, device=device, range=(-20.0, 20.0), title=f"Reconstruction: C2, k={k}, sig={sigma}, {modelType} Wits Preinit")


        # individual_functions_c1 = individual_kanlayers(act_fun_c1)
        # individual_functions_c2 = individual_kanlayers(act_fun_c2)
        # if PLOT_GRAPHS:
        #     for func in individual_functions_c1:
        #         plot_sympy_func(func, (-20.0, 20.0))
        #     for func in individual_functions_c2:
        #         plot_sympy_func(func, (-20.0, 20.0))
            
        composed_function_c1 = compose_kanlayers(act_fun_c1)
        composed_function_c2 = compose_kanlayers(act_fun_c2)
        print("COMPOSED FUNC:", composed_function_c1)
        print("COMPOSED FUNC:", composed_function_c2)

        if PLOT_GRAPHS:
            plot_sympy_func(composed_function_c1, (-20.0, 20.0))
            plot_sympy_func(composed_function_c2, (-20.0, 20.0))
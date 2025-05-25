import torch
import TrainingFramework
import TestingFramework
from DataVis import *
from SplineComposer import *
import kan
import sys
import config

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_device(device=device)

DISPLAY_HEATMAP = False
DISPLAY_SYMBOLIC = False

PLOT_GRAPHS = False

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("Require command line input: test for testing, train for training \n ie python main.py test \n Configure training or testing in config.py")
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if mode.lower() == "train":
        for c in config.configs:
            f = TrainingFramework.TrainingFramework(k_range=c["kvals"], sigma_range=c["sigvals"], KAN_hyps=c["kanHyps"])
            f.train_framework(c["kanType"], c["env"], c["trainer"], c["modelType"], prefit_func_1=c["prefit_func_1"], prefit_func_2=c["prefit_func_2"], lr=c["lr"])
    elif mode.lower() == "test":
        for c in config.testConfigs:
            f = TestingFramework.r2_dir(c["path"])
            f = TestingFramework.test_dir(c["path"], c["env"])
    else:
        pass
    # if TRAIN_DGDCOMB:
    #     kanType = CombinedKan.CombinedKan
    #     env = WitsEnv.WitsEnvCombined
    #     gradDesc = WitsPPO.WitsGradDescCombined
    #     modelType = 'DGDCOMB'
    #     f.train_framework(kanType, env, gradDesc, modelType)
    # if TRAIN_ALTERNATING:
    #     kanType = kan.KAN
    #     env = WitsEnv.WitsEnv
    #     gradDesc = WitsPPO.WitsAlternatingDescent
    #     modelType = "ALTERNATING"
    #     f.train_framework(kanType, env, gradDesc, modelType)

    if DISPLAY_HEATMAP:
        hyps =  [[1,0],[12,0],[1,0]]
        kvals, sigvals, losses = getLosses(dgd=False, dgdcomb=False, ppo=False, lag=False, lsa=False, fgd=True, hyps=hyps, models_loss_dir="./FGD_loss/")

        kvals_squared = [round(k**2/0.05)*0.05 for k in kvals]
        varvals = [round(s**2/5.0)*5.0 for s in sigvals]
        losses = [min(2.0, x) for x in losses]

        # lookup = generateLookupTable(kvals, sigvals, losses)
        # print(lookup[(0.3, 4.6)])

        modelType = 'FGD'

        create_heatmap(kvals_squared, varvals, losses, cmap='plasma', title=f"{modelType} {[x[0] for x in hyps]} Model Costs")

    if DISPLAY_SYMBOLIC:
        hyps = [[1,0],[20,0],[1,0]]

        # LINEAR (UNINTENTIONAL - SHOULD BE 3-STEP)
        k = "0.20"
        sigma = "5.00"

        # 3 STEP: TBF
        # k = 0.22
        # sigma = "5.00"

        # 5-STEP (INTENTIONAL)
        # k = 0.39
        # sigma = 3.87

        modelType = 'MOML'
        
        name = f"momentum_models/{modelType}-k-{k}-sigma-{sigma}-hyps-{hyps}-"
        actor_c1 = kan.KAN.loadckpt(name+"c1")
        actor_c2 = kan.KAN.loadckpt(name+"c2")
        act_fun_c1 = actor_c1.act_fun
        act_fun_c2 = actor_c2.act_fun
        
        if PLOT_GRAPHS:
            # actor_c1.plot()
            # plt.show()
            # actor_c2.plot()
            # plt.show()
        
            plot_model_bruteforce(actor_c1, device=device, range=(-15.0, 15.0), title=f"Reconstruction: C1, hyps={[x[0] for x in hyps]}, k={k}, sig={sigma}, {modelType}")
            plot_model_bruteforce(actor_c2, device=device, range=(-15.0, 15.0), title=f"Reconstruction: C2, hyps={[x[0] for x in hyps]}, k={k}, sig={sigma}, {modelType}")


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
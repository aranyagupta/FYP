import torch
import TrainingFramework
from DataVis import *
from SplineComposer import *

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_device(device=device)

TRAIN_DGD = False
TRAIN_PPO = False
TRAIN_DGDCOMB = False

DISPLAY_HEATMAP = True
DISPLAY_SYMBOLIC = True

PLOT_GRAPHS = False

if __name__ == "__main__":
    f = TrainingFramework.TrainingFramework()
    if TRAIN_DGD:
        f.train_dgd()
    if TRAIN_PPO:
        f.train_ppo()
    if TRAIN_DGDCOMB:
        f.train_dgd_combined()

    if DISPLAY_HEATMAP:
        kvals, sigvals, losses = getLosses(dgd=True, dgdcomb=False, ppo=False)

        kvals_squared = [k**2 for k in kvals]
        varvals = [s**2 for s in sigvals]
        losses = [min(2.0, x) for x in losses]

        # lookup = generateLookupTable(kvals, sigvals, losses)
        # print(lookup[(0.3, 4.6)])

        create_heatmap(kvals_squared, varvals, losses, cmap='plasma', title="DGD Model Costs")
    
    if DISPLAY_SYMBOLIC:
        actor_c1 = kan.KAN.loadckpt("wits_models/DGD-k-0.30-sigma-4.35-hyps-[[1, 0], [2, 0], [2, 0], [1, 0]]-c1")
        actor_c2 = kan.KAN.loadckpt("wits_models/DGD-k-0.30-sigma-4.35-hyps-[[1, 0], [2, 0], [2, 0], [1, 0]]-c2")

        act_fun_c1 = actor_c1.act_fun
        act_fun_c2 = actor_c2.act_fun
        
        individual_funcs_c1 = individual_kanlayers(act_fun_c1)
        individual_funcs_c2 = individual_kanlayers(act_fun_c2)
        print("C1 FUNCS:", individual_funcs_c1)
        print("C2 FUNCS:", individual_funcs_c2)

        if PLOT_GRAPHS:
            for func in individual_funcs_c1:
                plot_sympy_func(func, (-20.0, 20.0))

        composed_function_c1 = compose_kanlayers(act_fun_c1)
        composed_function_c2 = compose_kanlayers(act_fun_c2)
        print("COMPOSED FUNC:", composed_function_c1)
        print("COMPOSED FUNC:", composed_function_c2)

        if PLOT_GRAPHS:
            plot_sympy_func(composed_function, (-20.0, 20.0))
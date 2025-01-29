import torch
import TrainingFramework
from DataVis import *
from SplineComposer import *
import kan
from itertools import product

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_device(device=device)

TRAIN_DGD = True
TRAIN_PPO = False
TRAIN_DGDCOMB = False

DISPLAY_HEATMAP = False
DISPLAY_SYMBOLIC = False

PLOT_GRAPHS = False

if __name__ == "__main__":
    min_hidden_layers = 3
    max_hidden_layers = 10
    min_layer_width = 2
    max_layer_width = 8

    kvals = [0.05, 0.6, 0.05]
    sigvals = [2.0, 7.0, 0.5]
    kan_hyps = []
    for num_layers in range(min_hidden_layers, max_hidden_layers+1):
        for layer_width in range(min_layer_width, max_layer_width+1):
            hidden = [layer_width] * num_layers
            hyps = [1] + hidden + [1]
            kan_hyps.append(hyps)

    f = TrainingFramework.TrainingFramework(k_range=kvals, sigma_range=sigvals, KAN_hyps=kan_hyps)
    if TRAIN_DGD:
        f.train_dgd()
    if TRAIN_PPO:
        f.train_ppo()
    if TRAIN_DGDCOMB:
        f.train_dgd_combined()

    if DISPLAY_HEATMAP:
        kvals, sigvals, losses = getLosses(dgd=True, dgdcomb=False, ppo=False, hyps=[[1,0],[2,0],[2,0],[1,0]])

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
        
        if PLOT_GRAPHS:
            actor_c1.plot()
            plt.show()
            actor_c2.plot()
            plt.show()

        composed_function_c1 = compose_kanlayers(act_fun_c1)
        composed_function_c2 = compose_kanlayers(act_fun_c2)
        print("COMPOSED FUNC:", composed_function_c1)
        print("COMPOSED FUNC:", composed_function_c2)

        if PLOT_GRAPHS:
            plot_sympy_func(composed_function_c1, (-20.0, 20.0))
            plot_sympy_func(composed_function_c2, (-20.0, 20.0))
import torch
import kan
import WitsEnv
import WitsPPO
"""
Training configurations
"""
configs = [
    {
        "kvals":[0.22, 0.32, 0.39, 0.45, 0.50, 0.55, ],
        "sigvals":[2.24, 3.16, 3.87, 4.47, 5.00, 5.48, 5.92, 6.32],
        "kanHyps":[[1,10,1],[1,11,1],[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvFGD,
        "trainer":WitsPPO.WitsFGD,
        "modelType":"LSA",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": None,
        "lr":0.01,
    }
]

"""
Testing configurations
"""
testConfigs = [
    {
        "path":"FGD_models/",
        "env":WitsEnv.WitsEnvFGD
    }
]

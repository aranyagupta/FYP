import torch
import kan
import WitsEnv
import WitsPPO
"""
Training configurations
"""
configs = [
    {
        "kvals":[0.1, 0.2, 0.3, 0.4, 0.5],
        "sigvals":[5.0],
        "kanHyps":[[1,10,1],[1,11,1],[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvLSA,
        "trainer":WitsPPO.WitsLSA,
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

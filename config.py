import torch
import kan
import WitsEnv
import WitsPPO
"""
Training configurations
"""
configs = [
    {
        "kvals":[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "sigvals":[5.0],
        "kanHyps":[[1,20,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvFGD,
        "trainer":WitsPPO.WitsFGD,
        "modelType":"FGD",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": lambda sigma, x : sigma*torch.tanh(sigma*x),
        "lr":0.01,
    },
    {
        "kvals":[0.77],
        "sigvals":[2.2361, 3.1623, 3.8730, 4.4721, 5.0000, 5.4772, 5.9161, 6.3246, 6.7082],
        "kanHyps":[[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnv,
        "trainer":WitsPPO.WitsGradDesc,
        "modelType":"DGD",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": lambda sigma, x : sigma*torch.tanh(sigma*x),
        "lr":0.01,
    },
    {
        "kvals":[0.77],
        "sigvals":[2.2361, 3.1623, 3.8730, 4.4721, 5.0000, 5.4772, 5.9161, 6.3246, 6.7082],
        "kanHyps":[[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvFGD,
        "trainer":WitsPPO.WitsFGD,
        "modelType":"FGD",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": lambda sigma, x : sigma*torch.tanh(sigma*x),
        "lr":0.01,
    },
    {
        "kvals":[0.5477],
        "sigvals":[5.0000],
        "kanHyps":[[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvConstrained,
        "trainer":WitsPPO.WitsGradDescConstrained,
        "modelType":"LAG",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": lambda sigma, x : sigma*torch.tanh(sigma*x),
        "lr":0.01,
    },
    {
        "kvals":[0.77],
        "sigvals":[2.2361, 3.1623, 3.8730, 4.4721, 5.0000, 5.4772, 5.9161, 6.3246, 6.7082],
        "kanHyps":[[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvLSA,
        "trainer":WitsPPO.WitsLSA,
        "modelType":"LSA",
        "prefit_func_1": lambda sigma, x : sigma*torch.sign(x),
        "prefit_func_2": None,
        "lr":0.01,
    },

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

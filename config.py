import kan
import WitsEnv
import WitsTrainer
"""
Training configurations
"""
configs = [
    {
        "kvals":[0.20],
        "sigvals":[5.00],
        "kanHyps":[[1,20,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvMomentum,
        "trainer":WitsTrainer.WitsMomentum,
        "modelType":"MOML",
        "prefit_func_1": lambda x : x,
        "prefit_func_2": lambda x : x,
        "lr":0.01,
    },
    # Add more here...
]

"""
Testing configurations
"""
testConfigs = [
    {
        "path":"momentum_models/",
        "env":WitsEnv.WitsEnvMomentum
    },
    # Add more here...
]

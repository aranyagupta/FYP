import kan
import WitsEnv
import WitsTrainer
"""
Training configurations
"""
configs = [
    {
        "kvals":[1.00],
        "sigvals":[2.24, 3.16, 3.87, 4.47, 5.00, 5.48, 5.92, 6.32, 6.71],
        "kanHyps":[[1,12,1]],
        "kanType":kan.KAN,
        "env":WitsEnv.WitsEnvMomentum,
        "trainer":WitsTrainer.WitsMomentum,
        "modelType":"MOML",
        "prefit_func_1": None,
        "prefit_func_2": None,
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

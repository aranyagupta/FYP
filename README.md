# Towards an analytic solution to the Witsenhausen Problem using Kolmogorov-Arnold Networks

This is the main repository for the Final Year Project submitted as a part of the degree requirements for MEng Electronic and Information Engineering at Imperial College London.

Repository written and maintained by Aranya Gupta.

## User Guide


### Dependency installation
- Ensure Python 3.12.3 or above is installed
- Create a virtual environment with `python -m venv venv`
- Activate the environment with `source venv/bin/activate`
- Install dependencies with `pip install -r requirements.txt`
- Create necessary directories for training with `mkdir wits_models wits_models_loss`

### Configuration
In `config.py`, set up a configuration for the types of training you want to do. We show below the format for the dictionary for each Training session.

|      Key      | Allowed Types |                        Example Value                        |                                                                                                                             Explanation                                                                                                                             |
|:-------------:|:-------------:|:-----------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     kvals     |   float list  |                  [0.1, 0.2, 0.3, 0.4, 0.5]                  |                                                                        A set of values of k for which models should be trained. Models will be trained under all sigvals for a given k value.                                                                       |
|    sigvals    |   float list  |                   [5.0, 10.0, 15.0, 20.0]                   |                                                                        A set of values of σx for which models should be trained. Models will be trained under all kvals for a given σx value.                                                                       |
|    kanHyps    | int list list |                 [[1,10,1],[1,11,1],[1,12,1]]                |                       A list of hyperparameters determining the structure of each KAN to be trained. Each element of kanHyps describes one KAN hy- perparameter value, and should begin and end with 1 (as this is a 1 dimensional prob- lem).                      |
|    kanType    |    kan.KAN    |                           kan.KAN                           |                                                                             The type of machine learning paradigm to be trained. This is deprecated, and should only be set to kan.KAN.                                                                             |
|      env      |  WitsEnvSuper |                          WitsEnvLSA                         |                                                    The learning environment in which the KAN is to be trained. Must match the correspond- ing Trainer type. Any object that derives the Allowed Type is allowed.                                                    |
|    trainer    |  WitsTrainer  |                           WitsLSA                           |                                                                 The Trainer object for the KAN. Must match the corresponding Environment Type. Any object that derives the Allowed Type is allowed.                                                                 |
|   modelType   |     String    |                             LSA                             |                                                                                 Acronym for model storage. We recommend a three letter acronym that describes the training process.                                                                                 |
| prefit_func_1 |  lambda/None  | ambda x : torch.floor(x) lambda s, x : s*torch.sign(x) None | The function f(x) to which the first con- troller is to be prefit to, if required. Can take in one or two arguments, the final of which should be the actual argument for f. If two arguments are used, the first can be used to modify the function (see example). |
| prefit_func_2 |  lambda/None  |                           As above                          |                                                                                            As above, but prefitting to this function is applied to the second controller.                                                                                           |
|       lr      |     float     |                             0.01                            |                                                                                          The learning rate for the optimisers for each KAN. We recommend a value of ∼0.01.                                                                                          |
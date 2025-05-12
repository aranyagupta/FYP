import kan
import os
import torch

# given a file name, extract the core file name 
# ie ABC-k-x.xx-sigma-x.xx-hyps-[x,x,x]
# and k and sigma as a pair of values
def extract_info(file):
    try:
        i = file.index("-c1")
    except:
        i = 0

    try:
        k = float(file[6:10])
        sigma = float(file[17:21])
    except:
        k = -1
        sigma = -1
    return file[:i], k, sigma
            


# given a path and an environment type, find and test all KAN models in that environment 
def test_dir(path, env):
    print(f"Running test suite on dir {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path[-1] != "/":
        path = path + "/"
    files = os.listdir(path)

    core_names = []
    kvals = []
    sigmavals = []

    for file in files:
        c, k, s = extract_info(file)
        core_names.append(c)
        kvals.append(k)
        sigmavals.append(s)

    for i in range(len(core_names)):
        if core_names[i] != "":
            c1 = kan.KAN.loadckpt(path+core_names[i]+"-c1")
            c2 = kan.KAN.loadckpt(path+core_names[i]+"-c2")
            k = kvals[i]
            s = sigmavals[i]
            testEnv = env(k, s, device, mode="TEST")
            rew = testEnv.step_timesteps(c1, c2, timesteps=100000)
            print("Tested:", core_names[i], "value:", rew)


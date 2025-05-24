import kan
import os
import torch
import numpy as np

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
    return file[:i], abs(k), abs(sigma)
            

def calculate_r2(actor, device, range=(-15.0, 15.0)):
    num_points = 1000
    x_values = torch.linspace(range[0], range[1], num_points).to(device)
    x_values_np = x_values.cpu().detach().numpy()
    x_values = x_values.reshape(num_points, 1)

    y_values = actor(x_values)
    y_values_np = y_values.reshape(num_points)
    y_values_np = y_values_np.cpu().detach().numpy()

    m, b = np.polyfit(x_values_np, y_values_np, 1)
    y_pred = m*x_values_np + b

    ss_res = np.sum((y_values_np-y_pred)**2)
    ss_tot = np.sum((y_values_np-np.mean(y_values_np))**2)
    rs = 1.0 - (ss_res / ss_tot)

    return rs, m, b

def r2_dir(path):
    print(f"Running R^2 calculations on dir {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path[-1] != "/":
        path = path + "/"
    files = os.listdir(path)

    core_names = []
    sigvals = []
    for file in files:
        c, _, s = extract_info(file)
        core_names.append(c)
        sigvals.append(s)

    seen_before = {}    
    for i in range(len(core_names)):
        try:
            seen = seen_before[core_names[i]]
        except:
            seen = False
        if core_names[i] != "" and not seen:
            c1 = kan.KAN.loadckpt(path+core_names[i]+"-c1")
            c2 = kan.KAN.loadckpt(path+core_names[i]+"-c2")
            r2_c1, m1, b1 = calculate_r2(c1, device, (-3*sigvals[i], 3*sigvals[i]))
            r2_c2, m2, b2 = calculate_r2(c2, device, (-3*sigvals[i], 3*sigvals[i]))
            print(f"{core_names[i]}-c1 R^2 value: {r2_c1}, m: {m1}, b: {b1}")
            print(f"{core_names[i]}-c2 R^2 value: {r2_c2}, m: {m2}, b: {b2}")

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
        # kvals.append(0.2)
        # sigmavals.append(5.0)

    seen_before = {}    
    count = 0
    for i in range(len(core_names)):
        try:
            seen = seen_before[core_names[i]]
        except:
            seen = False
        if core_names[i] != "" and not seen:
            c1 = kan.KAN.loadckpt(path+core_names[i]+"-c1")
            c2 = kan.KAN.loadckpt(path+core_names[i]+"-c2")
            k = kvals[i]
            s = sigmavals[i]
            testEnv = env(k, s, device, mode="TEST")
            rew = testEnv.step_timesteps(c1, c2, timesteps=100000)
            print("Tested:", core_names[i], "value:", rew)
            seen_before[core_names[i]] = 1
            count +=1
    print("TESTED", count, "MODELS")


import matplotlib.pyplot as plt
import os
import numpy as np

def get_loss_from_file(filename, models_loss_dir="./wits_models_loss_storage/"):
    file = open(models_loss_dir+filename)
    loss_text = file.readlines()[-1]
    loss_text = float(loss_text[11:])
    return loss_text

def getLosses(dgd=True, dgdcomb=False, ppo=False, lag=False, lsa=False, fgd=False, moml=False, hyps=[[1,0],[2,0],[2,0],[1,0]], models_loss_dir="./wits_models_loss_storage/"):
    files = os.listdir(models_loss_dir)
    if not dgdcomb:
        files = [x for x in files if x[:7]!="DGDCOMB"]
    if not dgd:
        files = [x for x in files if x[:3]!="DGD"]
    if not ppo:
        files = [x for x in files if x[:3]!="PPO"]
    if not lag:
        files = [x for x in files if x[:3]!="LAG"]
    if not lsa:
        files = [x for x in files if x[:3]!="LSA"]
    if not fgd:
        files = [x for x in files if x[:3]!="FGD"]
    if not moml:
        files = [x for x in files if x[:4]!="MOML"]

    files = [file for file in files if str(hyps) in file]

    k_start = 6
    k_end = 10
    sig_start = 17
    sig_end = 21

    kvals = [float(x[k_start:k_end]) for x in files]
    sigvals = [float(x[sig_start:sig_end]) for x in files]
    losses = [get_loss_from_file(file, models_loss_dir=models_loss_dir) for file in files]
    return kvals, sigvals, losses

def create_heatmap(x_values, y_values, z_values, cmap='viridis', title="Heatmap"):
    # Create a meshgrid of x and y values
    x_unique = np.unique(x_values)
    y_unique = np.unique(y_values)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)

    # Ensure at least 2 unique values in each axis for proper heatmap plotting
    if len(x_unique) == 1:
        x_unique = np.append(x_unique, x_unique[0] + 1)
    if len(y_unique) == 1:
        y_unique = np.append(y_unique, y_unique[0] + 1)

    # Create a 2D array for z values
    
    z_grid = np.full(x_grid.shape, np.nan)
    for x, y, z in zip(x_values, y_values, z_values):
        i = np.where(y_unique == y)[0][0]
        j = np.where(x_unique == x)[0][0]
        z_grid[i, j] = z

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(z_grid, origin='lower', cmap=cmap,
               extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
               aspect='auto')
    plt.colorbar(label='Test Cost')
    plt.title(title)
    plt.xlabel('k^2')
    plt.ylabel('sigma^2')
    plt.show()

def generateLookupTable(kvals, sigvals, losses):
    assert len(kvals) == len(sigvals)
    assert len(losses) == len(kvals)

    lookup = {}
    for i in range(len(kvals)):
        lookup[(kvals[i], sigvals[i])] = losses[i]
    
    return lookup
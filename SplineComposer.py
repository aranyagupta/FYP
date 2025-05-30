import torch
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
import time

def symbolic_b_splines_sum(coef, k, knots):
    """
    Generate a symbolic representation of the sum of B-splines of order k.

    Parameters:
        coef (np.ndarray): Coefficients of the B-splines.
        k (int): Order of the B-splines.

    Returns:
        sp.Expr: Symbolic representation of the sum of B-splines.
    """
    # Ensure coefficients are in a numpy array for iteration
    coef = np.asarray(coef)

    # Number of basis functions (length of coefficients)
    num_b_splines = len(coef)

    # Create a symbolic variable for the parameter t
    t = sp.symbols('t')

    # Generate the knots vector
    knots = np.array(knots)

    print(num_b_splines, len(knots))

    # Order k - for some reason, the authors of the KAN say 
    # an order k spline uses order k polynomials. This is not
    # actually true - an order k spline uses order p=k-1 polynomials. 
    # ie an order 3 B-spline uses quadratic polynomials, not cubic
    # since the authors have implemented it incorrectly, we need to compensate 
    # and set p=k when finding the b-splines. 
    p = k 

    memo = {}

    # Function to recursively calculate B-splines basis
    def b_spline_basis(i, p, t, knots):
        if p == 0:
            memo[(i, p)] = sp.Piecewise((1, (t >= knots[i]) & (t < knots[i + 1])), (0, True))
            return memo[(i, p)]
        else:
            try:
                s = memo[(i, p)]
            except: 
                denom1 = knots[i + p] - knots[i]
                term1 = ((t - knots[i]) / denom1 * b_spline_basis(i, p - 1, t, knots)) if denom1 != 0 else 0
                term1 = sp.simplify(term1)

                denom2 = knots[i + p + 1] - knots[i + 1]
                term2 = ((knots[i + p + 1] - t) / denom2 * b_spline_basis(i + 1, p - 1, t, knots)) if denom2 != 0 else 0
                term2 = sp.simplify(term2)
                memo[(i, p)] = sp.simplify(term1+term2)
                s = memo[(i, p)]
            

            # return term1 + term2
            return s

    # Construct the symbolic sum of B-splines weighted by coefficients
    # spline_sum = sum(coef[i] * b_spline_basis(i, p, t, knots) for i in range(num_b_splines)) # SLOW - DO NOT USE

    spline_sum = 0
    for i in range(num_b_splines):
        start = time.time()
        spline_sum = spline_sum + coef[i] * b_spline_basis(i, p, t, knots)
        if i != 0:
            spline_sum = sp.simplify(spline_sum)
        print(f"finished simplifing B-spline sum up to i={i}, took {time.time()-start}s")

    # return sp.simplify(spline_sum)
    return spline_sum


# Composes kan layers found in actor.act_fun into one symbolic function
def compose_kanlayers(act_fun):
    t = sp.symbols('t')
    initial_funcs = [t]
    for layer_i in range(len(act_fun)):
        print("ON LAYER:", layer_i)
        layer = act_fun[layer_i]
        order = layer.k
        out_funcs = [0 for _ in range(layer.out_dim)]
        for j in range(layer.out_dim):
            for i in range(layer.in_dim):
                print("on i, j:", i, j)
                coefficients = layer.coef[i][j].cpu().detach().numpy()
                print("here 1")
                knots = layer.grid[i].cpu().detach().numpy()
                print("here 2")

                func = symbolic_b_splines_sum(coefficients, order, knots)
                print("here 3")
                func = layer.scale_sp[i, j]*func + layer.scale_base[i, j] * t/(1+sp.exp(-t))
                print("here 4")
                start = time.time()
                out_funcs[j] += func.subs({"t":initial_funcs[i]})
                print(f"finished substitution, took {time.time()-start}s")

        initial_funcs = out_funcs
    assert len(initial_funcs) == 1, "initial_funcs does not have length 1, something went wrong"
    return initial_funcs[0]

# returns individual kan layers functions found in actor.act_fun
# if actor is given, plots individual kanlayer functions on same plot as actor
# plots the same way it's done in kan.plot()
def individual_kanlayers(act_fun, actor=None):
    t = sp.symbols('t')
    function_list = []
    for l in range(len(act_fun)):
        layer = act_fun[l]
        order = layer.k
        for i in range(layer.in_dim):
            for j in range(layer.out_dim):
                coefficients = layer.coef[i][j].cpu().detach().numpy()
                knots = layer.grid[i].cpu().detach().numpy()
                
                func = symbolic_b_splines_sum(coefficients, order, knots)
                func = layer.scale_sp[i, j]*func + layer.scale_base[i, j] * t/(1+sp.exp(-t))
                function_list.append(func)

    if not (actor is None):
        depth = len(actor.width) - 1
        for l in range(depth):
            for i in range(actor.width_in[l]):
                for j in range(actor.width_out[l+1]):
                    rank = torch.argsort(actor.acts[l][:, i])
                    sp_func = function_list[(depth+1)*l + i+j] 
                    interval = [-15.0, 15.0]
                    num_points = 1000
                    t_values = np.linspace(interval[0], interval[1], num_points)
                    try:
                        spline_function = lambdify(t, sp_func, 'numpy')
                        y_values = spline_function(t_values)
                    except:
                        spline_function = lambdify(t, sp_func, 'sympy')
                        y_values = []

                        for t in t_values:
                            y_values.append(spline_function(t))

                    plt.plot(actor.acts[l][:, i][rank].cpu().detach().numpy(), actor.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), lw=5, label="Computational output")
                    plt.plot(t_values, y_values, label=f"Symbolic B-spline output")
                    plt.xlabel("t", fontsize=12)
                    plt.ylabel("f(t)", fontsize=12)
                    plt.title("Symbolic and Computational comparison", fontsize=14)
                    plt.grid(True)
                    plt.legend(fontsize=12)
                    plt.show()

    return function_list

# plots a given sympy function
def plot_sympy_func(sympy_func, interval, num_points=1000):
    # Generate symbolic B-spline sum
    symbolic_spline = sympy_func

    # Create a numerical function from the symbolic representation
    t = sp.symbols('t')
    t_values = np.linspace(interval[0], interval[1], num_points)
    # we might not be able to use numpy's vectorisation if there are 
    # if-then-else statements in the function - if there are, we do it 
    # much more slowly under the sympy lambdify backend
    try:
        spline_function = lambdify(t, symbolic_spline, 'numpy')
        y_values = spline_function(t_values)
    except:
        spline_function = lambdify(t, symbolic_spline, 'sympy')
        y_values = []

        for t in t_values:
            y_values.append(spline_function(t))

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, label=f"Sum of B-splines")
    plt.xlabel("t", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Plot of Sum of B-splines", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

# Brute-force plots the model output over a range of values
# without computing the symbolic representation (for quick results)
def plot_model_bruteforce(actor, device, range=(-15.0, 15.0), title="Reconstruction: C1 LAG"):
    num_points = 1000
    x_values = torch.arange(range[0], range[1], (range[1]-range[0])/float(num_points)).to(device)
    x_values_plot = x_values.cpu().detach().numpy()
    x_values = x_values.reshape(num_points, 1)

    y_values = actor(x_values)
    y_values_plot = y_values.reshape(num_points)
    y_values_plot = y_values_plot.cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x_values_plot, y_values_plot, label=f"f(x)")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.title(f"{title}", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

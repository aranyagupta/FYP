import kan
import torch
from kan.utils import ex_round
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify

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

    # Function to recursively calculate B-splines basis
    def b_spline_basis(i, k, t, knots):
        if k == 1:
            return sp.Piecewise((1, (t >= knots[i]) & (t < knots[i + 1])), (0, True))
        else:
            denom1 = knots[i + k - 1] - knots[i]
            term1 = ((t - knots[i]) / denom1 * b_spline_basis(i, k - 1, t, knots)) if denom1 != 0 else 0

            denom2 = knots[i + k] - knots[i + 1]
            term2 = ((knots[i + k] - t) / denom2 * b_spline_basis(i + 1, k - 1, t, knots)) if denom2 != 0 else 0

            return term1 + term2

    # Construct the symbolic sum of B-splines weighted by coefficients
    spline_sum = sum(coef[i] * b_spline_basis(i, k, t, knots) for i in range(num_b_splines))

    # return sp.simplify(spline_sum)
    return spline_sum


# Composes kan layers found in actor.act_fun into one symbolic function
def compose_kanlayers(act_fun):
    t = sp.symbols('t')
    initial_funcs = [t]
    for layer in act_fun:
        order = layer.k
        out_funcs = [0 for _ in range(layer.out_dim)]
        for j in range(layer.out_dim):
            for i in range(layer.in_dim):
                coefficients = layer.coef[i][j].cpu().detach().numpy()
                knots = layer.grid[i].cpu().detach().numpy()

                func = symbolic_b_splines_sum(coefficients, order, knots)
                func = layer.scale_sp[i, j]*func + layer.scale_base[i, j] * t/(1+sp.exp(-t))
                out_funcs[j] += func.subs({"t":initial_funcs[i]})
        initial_funcs = out_funcs

    return initial_funcs[0]

# Plots and returns individual kan layers functions found in actor.act_fun
def individual_kanlayers(act_fun):
    t = sp.symbols('t')
    function_list = []
    for layer in act_fun:
        order = layer.k
        for i in range(layer.in_dim):
            for j in range(layer.out_dim):
                coefficients = layer.coef[i][j].cpu().detach().numpy()
                knots = layer.grid[i].cpu().detach().numpy()
                
                plot_interval = (knots[0], knots[-1])
                func = symbolic_b_splines_sum(coefficients, order, knots)
                func = layer.scale_sp[i, j]*func + layer.scale_base[i, j] * t/(1+sp.exp(-t))
                function_list.append(func)

    return function_list

def plot_sympy_func(sympy_func, interval, num_points=1000):
    """
    Plot the function represented by the sum of B-splines.

    Parameters:
        coef (np.ndarray): Coefficients of the B-splines.
        k (int): Order of the B-splines.
        interval (tuple): The range (start, end) over which to plot the function.
        num_points (int): Number of points to use for plotting.

    Returns:
        None
    """
    # Generate symbolic B-spline sum
    symbolic_spline = sympy_func

    # Create a numerical function from the symbolic representation
    t = sp.symbols('t')
    spline_function = lambdify(t, symbolic_spline, 'numpy')

    # Generate t values in the interval
    t_values = np.linspace(interval[0], interval[1], num_points)

    # Evaluate the spline function at those points
    y_values = spline_function(t_values)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, label=f"Sum of B-splines")
    plt.xlabel("t", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Plot of Sum of B-splines", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
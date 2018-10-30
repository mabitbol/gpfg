import numpy as np
import sympy as sym

hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS

def sym_power_law():
    amp, beta, x, x_0 = sym.symbols('amp beta x x_0')
    expr = amp * sym.power.Pow(x/x_0, beta)
    return [amp, beta, x, x_0], expr

def power_law_expansion(nmoments=0):
    [amp, beta, x, x_0], expr = sym_power_law()
    params = [amp, beta]
    specfncs = [expr]
    for k in range(nmoments):
        specfncs_temp = []
        specfncs_temp.append(specfncs[k].diff(amp))
        specfncs_temp.append(specfncs[k].diff(beta))
        specfncs += list(set(specfncs_temp))
    moments = sym.symarray('w', len(specfncs)-1)
    return params, moments, specfncs

def eval_power_law(nu, amp_0=288., beta_0=-0.82, nu_0=100.e9):
    [amp, beta, x, x_0], expr = sym_power_law()
    expr = expr.subs([(amp, amp_0), (beta, beta_0), (x_0, nu_0)])
    eval_expr = sym.lambdify(x, expr, "numpy")
    return eval_expr(nu)


def sym_mbb(nmoments=0):
    amp, beta, temp, x, X = sym.symbols('amp beta temp x X')
    X = hplanck * x / (kboltz * temp)
    expr = amp * sym.power.Pow(X, beta) * X**3 / (sym.exp(X) - 1. )
    return [amp, beta, temp, x], expr


def eval_mbb(nu, amp_0=1.36e6, beta_0=1.53, temp_0=21.):
    [amp, beta, temp, x], expr = sym_mbb()
    expr = expr.subs([(amp, amp_0), (beta, beta_0), (temp, temp_0)])
    eval_expr = sym.lambdify(x, expr, "numpy")
    return eval_expr(nu)

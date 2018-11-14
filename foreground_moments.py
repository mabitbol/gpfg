import sympy as sym
import numpy as np

hplanck = 6.626068e-34
kboltz = 1.3806503e-23

def sym_power_law():
    x_0 = 100.e9
    x, amp, beta = sym.symbols('x amp beta')
    expr = amp * sym.power.Pow(x/x_0, beta)
    params = [amp, beta]
    return x, [amp, beta], expr

def sym_mbb():
    x, amp, beta, invtemp = sym.symbols('x amp beta invtemp')
    X = hplanck * x * invtemp / kboltz
    expr = amp * sym.power.Pow(X, beta) * X**3 / (sym.exp(X) - 1. )
    return x, [amp, beta, invtemp], expr

def moment_derivatives(args, expr_list, order):
    if order:
        diff_list = [expr.diff(arg) for expr in expr_list for arg in args]
        return moment_derivatives(args, diff_list, order-1)
    return expr_list

def clean_derivatives(expr_list):
    expr_list = list(set(expr_list))
    while 0 in expr_list: expr_list.remove(0)
    return expr_list

def calculate_moment_expansion_trick(args, expr, order, delete_first_moment=False):    
    # sad hack
    expansion = expr
    
    index_start = 1
    if delete_first_moment:
        index_start = 2
    moment_indices = range(index_start, order+1)
    
    spectral_functions = []
    for k in moment_indices:
        deriv_list = moment_derivatives(args, [expr/args[0]], k)
        spectral_functions += clean_derivatives(deriv_list)
        
    moments = list(sym.symarray('w', len(spectral_functions)))
    if spectral_functions:
        expansion += sym.Matrix(moments).dot(sym.Matrix(spectral_functions))
    return moments, expansion

def moment_expansion(sym_function, order, 
                     delete_first_moment=False, print_expansion=False):
    x, args, expr = sym_function()
    moments, expansion = calculate_moment_expansion_trick(args, expr, order, delete_first_moment)
    params = [x] + args + moments
    if print_expansion:
        print(moments, expansion)
    return sym.lambdify(params, expansion, "numpy")

# need to be able to grab individual moments / spectral functions
# need to be able to mask individual moments

def power_spectra_fgs(nu1, nu2, bs, bd, moments):
    n1 = nu1 / 100.e9
    n2 = nu2 / 100.e9
    synch = (n1*n2)**bs * ( moments[0] + (np.log(n1) + np.log(n2)) * moments[1] + \
             np.log(n1) * np.log(n2) * moments[2])
    dust = (n1*n2)**bd * ( moments[3] + (np.log(n1) + np.log(n2)) * moments[4] + \
             np.log(n1) * np.log(n2) * moments[5]) * bbX(nu1) * bbX(nu2)
    mixed = (n1**bs * n2**bd * bbX(nu2) + n2**bs * n1**bd * bbX(nu1) ) * moments[6]
    return synch + dust + mixed

def bbX(nu, Td=20.):
    x = hplanck * nu / (kboltz * Td)
    return x**3 / (np.exp(x) - 1.)
    

##################################################################################
def calculate_moment_expansion_raw(args, expr, order):
    # doesn't do anything useful
    spectral_functions = []
    for k in range(order+1):
        deriv_list = moment_derivatives(args, [expr], k)
        spectral_functions += clean_derivatives(deriv_list)
        
    moments = list(sym.symarray('w', len(spectral_functions)))
    expansion = sym.Matrix(moments).dot(sym.Matrix(spectral_functions))
    return moments, expansion


import sympy as sym

hplanck = 6.626068e-34  # MKS
kboltz = 1.3806503e-23  # MKS

def sym_power_law():
    x_0 = 100.e9
    x, amp, beta = sym.symbols('x amp beta')
    expr = amp * sym.power.Pow(x/x_0, beta)
    params = [amp, beta]
    return x, [amp, beta], expr

def sym_mbb():
    x, amp, beta, temp = sym.symbols('x amp beta temp')
    X = hplanck * x / (kboltz * temp)
    expr = amp * sym.power.Pow(X, beta) * X**3 / (sym.exp(X) - 1. )
    return x, [amp, beta, temp], expr

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


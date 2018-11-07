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

def calculate_moment_expansion(args, expr, order, delete_first_moment=False):
    expansion = expr
    
    index_start = 1
    if delete_first_moment:
        index_start = 2
    moment_indices = range(index_start, order+1)

    spectral_functions = []
    for k in moment_indices:
        deriv_list = moment_derivatives(args, [expr], k)
        spectral_functions += clean_derivatives(deriv_list)
        
    if spectral_functions:
        moments = sym.symarray('w', len(spectral_functions))
        expansion = sym.Matrix(moments).dot(sym.Matrix(spectral_functions))
    else:
        moments = []
    return moments, expansion


###########################################################################


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

def eval_mbb(nu, amp_0=1.36e6, beta_0=1.53, temp_0=21.):
    [amp, beta, temp, x], expr = sym_mbb()
    expr = expr.subs([(amp, amp_0), (beta, beta_0), (temp, temp_0)])
    eval_expr = sym.lambdify(x, expr, "numpy")
    return eval_expr(nu)


## these work down here
def eval_power_law_expansion(nmoments):
    x, moments, specfncs = power_law_expansion(nmoments)
    expansion = sym.Matrix(moments).dot(sym.Matrix(specfncs))
    w_1 = sym.Symbol('w_1')
    expansion = expansion.subs(w_1, 0)
    
    params = list(expansion.free_symbols)
    params.remove(x)
    params = [x] + params
    print(params)
    print(sym.simplify(expansion))
    return sym.lambdify(params, expansion, "numpy")


def power_law_expansion(nmoments=0):
    x, [amp, beta], expr = sym_power_law()
    specfncs = [expr/amp]
    for k in range(nmoments):
        specfncs.append(specfncs[k].diff(beta)) 
    #specfncs = list(set(specfncs))
    moments = sym.symarray('w', len(specfncs))
    return x, moments, specfncs


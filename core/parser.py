import sympy as sp


def parse_and_reduce_ode(eq_string, dep_var_str="x", indep_var_str="t"):
    """
    Parses a 2nd-order ODE string and reduces it to a system of 1st-order ODEs.

    Example input: "d2x - (-4*x)"
    """
    t = sp.Symbol(indep_var_str)
    x = sp.Function(dep_var_str)(t)

    v = sp.Symbol("v")
    x_sym = sp.Symbol(dep_var_str)

    expr = sp.sympify(eq_string)

    d2x = sp.Symbol("d2x")
    expr_substituted = expr.subs(sp.Derivative(x, t, 2), d2x).subs(x, x_sym)
    sol_d2x = sp.solve(expr_substituted, d2x)[0]

    func_v = sp.lambdify((t, x_sym, v), v, "numpy")
    func_dv = sp.lambdify((t, x_sym, v), sol_d2x, "numpy")

    def system_rhs(t_val, state):
        pos, vel = state[0], state[1]
        return [func_v(t_val, pos, vel), func_dv(t_val, pos, vel)]

    return system_rhs
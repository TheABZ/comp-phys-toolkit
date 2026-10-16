import numpy as np

# Butcher Tableaus for RK2, RK4, and RK6
BUTCHER_TABLEAUS = {
    "RK2": {
        "c": np.array([0.0, 1.0]),
        "b": np.array([0.5, 0.5]),
        "A": np.array([[0.0, 0.0], [1.0, 0.0]]),
    },
    "RK4": {
        "c": np.array([0.0, 0.5, 0.5, 1.0]),
        "b": np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        "A": np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    },
    "RK6": {
        "c": np.array([0.0, 1.0, 0.5, 2 / 3, 0.2, 0.8, 1.0]),
        "b": np.array([1 / 9, 0.0, 9 / 20, 16 / 45, 49 / 180, 49 / 180, 0.0]),
        "A": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3 / 8, 1 / 8, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8 / 9, -8 / 9, 8 / 9, 0.0, 0.0, 0.0, 0.0],
                [3 / 10, -3 / 10, 3 / 10, 3 / 10, 0.0, 0.0, 0.0],
                [-4 / 5, 4 / 5, -4 / 5, 4 / 5, 4 / 5, 0.0, 0.0],
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0],
            ]
        ),
    },
}


def generalized_rk_step(func, t, y, h, method="RK4"):
    """Executes a single step for RK2, RK4, or RK6 using Butcher Tableaus."""
    tableau = BUTCHER_TABLEAUS[method]
    c, b, A = tableau["c"], tableau["b"], tableau["A"]
    stages = len(b)

    k = np.zeros((stages, len(y)))

    for i in range(stages):
        t_stage = t + c[i] * h
        y_stage = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
        k[i] = func(t_stage, y_stage)

    return y + h * np.sum(b[:, None] * k, axis=0)


def solve_ode_system(func, y0, t_span, dt, method="RK4"):
    """Integrates a vector ODE system across t_span."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t_eval)

    y_sol = np.zeros((n_steps, len(y0)))
    y_sol[0] = y0

    for i in range(1, n_steps):
        y_sol[i] = generalized_rk_step(
            func, t_eval[i - 1], y_sol[i - 1], dt, method=method
        )

    return t_eval, y_sol
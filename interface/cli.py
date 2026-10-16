import matplotlib.pyplot as plt

from core.parser import parse_and_reduce_ode
from core.rk_engine import solve_ode_system


def run_cli():
    print("====================================================")
    print("  COMPUTATIONAL PHYSICS MATHEMATICAL TOOLKIT")
    print("====================================================")
    print("Supported Schemes: RK2, RK4, RK6 (6th-Order Runge-Kutta)")
    print("----------------------------------------------------\n")

    eq_input = input(
        "Enter 2nd-Order ODE RHS [e.g. -4*x for harmonic oscillator]: "
    )
    if not eq_input.strip():
        eq_input = "-4*x"

    method = input("Select Solver Method (RK2, RK4, RK6) [Default: RK4]: ").upper()
    if method not in ["RK2", "RK4", "RK6"]:
        method = "RK4"

    x0 = float(input("Initial Position x(0) [Default: 1.0]: ") or 1.0)
    v0 = float(input("Initial Velocity v(0) [Default: 0.0]: ") or 0.0)
    t_end = float(
        input("Simulation End Time (seconds) [Default: 10.0]: ") or 10.0
    )
    dt = float(input("Time Step Size dt [Default: 0.05]: ") or 0.05)

    print(f"\n[Processing] Parsing expression and initializing {method} solver...")

    full_eq_str = f"d2x - ({eq_input})"
    sys_func = parse_and_reduce_ode(full_eq_str)

    y0 = [x0, v0]
    t, sol = solve_ode_system(
        sys_func, y0, (0.0, t_end), dt, method=method
    )

    print(f"[Complete] Integration successful ({len(t)} time steps computed).")

    plt.figure(figsize=(10, 5))
    plt.plot(t, sol[:, 0], label=f"Position x(t) [{method}]", color="crimson", lw=2)
    plt.plot(
        t,
        sol[:, 1],
        label=f"Velocity v(t) [{method}]",
        color="navy",
        linestyle="--",
        lw=1.5,
    )
    plt.xlabel("Time (t)")
    plt.ylabel("State Vector Value")
    plt.title(f"Numerical Solution of $x'' = {eq_input}$ using {method} Scheme")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = f"figures/solution_{method}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] Output plot saved to '{plot_path}'.")
    plt.show()
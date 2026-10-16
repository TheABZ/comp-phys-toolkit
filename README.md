# Generalized Computational Physics Mathematical Toolkit

A Python framework designed for symbolic ordinary differential equation (ODE) parsing, automated state-space order reduction, and high-precision numerical initial-value integration using generalized Butcher Tableau Runge-Kutta schemes (RK2, RK4, RK6).

## Theoretical Overview

### Symbolic ODE Reduction
The package parses arbitrary 2nd-order non-linear differential equations of the form:

$$\frac{d^2x}{dt^2} = f\left(t, x, \frac{dx}{dt}\right)$$

and dynamically maps them into coupled 1st-order state-space systems:

$$\begin{pmatrix} \dot{x} \\ \dot{v} \end{pmatrix} = \begin{pmatrix} v \\ f(t, x, v) \end{pmatrix}$$

### Generalized Butcher Tableau Integration
Rather than using fixed formulas, the integration engine uses stage evaluation over general Butcher Tableaus:

$$y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i$$

where intermediate stage derivatives $k_i$ are computed recursively:

$$k_i = f\left(t_n + c_i h, \; y_n + h \sum_{j=1}^{i-1} a_{ij} k_j\right)$$

Supported schemes include:
- **RK2**: 2nd-Order Midpoint / Explicit Runge-Kutta.
- **RK4**: Standard Classical 4th-Order Runge-Kutta.
- **RK6**: 6th-Order 7-Stage Luther/Runge-Kutta Integration Scheme.

## Project Architecture

```text
comp-phys-toolkit/
├── core/
│   ├── parser.py        # SymPy symbolic string parser & order reduction engine
│   └── rk_engine.py     # Unified Butcher Tableau vector solver (RK2, RK4, RK6)
├── interface/
│   └── cli.py           # Interactive command-line driver
├── figures/             # Exported trajectory and field plots
├── main.py              # Application entry point
└── requirements.txt     # Dependency specifications
"""
Starter code for the problem "Cart-pole swing-up".

Autonomous Systems Lab (ASL), Stanford University
"""

import time

from animations import animate_cartpole

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import odeint


def linearize(f, s, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    A, B = jax.jacobian(f, argnums=(0, 1))(s, u)
    ###########################################################################
    return A, B


def ilqr(f, s0, s_goal, N, Q, R, QN, eps=1e-4, max_iters=100000):
    """Compute the iLQR set-point tracking solution.
`
    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    QN : numpy.ndarray
        The terminal state cost matrix (2-D).
    eps : float, optional
        Termination threshold for iLQR.
    max_iters : int, optional
        Maximum number of iLQR iterations.

    Returns
    -------
    s_bar : numpy.ndarray
        A 2-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 2-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    Y : numpy.ndarray
        A 3-D array where `Y[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    y : numpy.ndarray
        A 2-D array where `y[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    if max_iters <= 1:
        raise ValueError("Argument `max_iters` must be at least 1.")
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize gains `Y` and offsets `y` for the policy
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))

    # Initialize the nominal trajectory `(s_bar, u_bar`), and the
    # deviations `(ds, du)`
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])
    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    count = 0
    for _ in range(max_iters):
        count += 1
        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B = np.array(A), np.array(B)

        # PART (c) ############################################################
        # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
        # raise NotImplementedError()

        # # From the lectur notes:
        # V - value function
        # V_x - gradient
        # V_xx - Hessian

        qN = 2 * QN @ (s_bar[-1] - s_goal)
        # value function and gradients
        V = np.zeros((N + 1, 1))
        V_x = np.zeros((N + 1, n))
        V_xx = np.zeros((N + 1, n, n))

        # cost function for nominal trajectory
        c_k = np.zeros((N, 1)) # initialize stage wise cost
        c_x = np.zeros((N, n)) 
        c_u = np.zeros((N, m))
        c_xx = np.zeros((N, n,n))
        c_uu = np.zeros((N, m,m))
        c_ux = np.zeros((N, m,n))

        # linear terms
        qk = np.zeros((N + 1, n))
        rk = np.zeros((N + 1, m))

        # c_k[-1] = 1/2 * (s_bar[-1] - s_goal).T @ QN @ (s_bar[-1] - s_goal)
        V[-1] = 1/2 * (s_bar[-1] - s_goal).T @ QN @ (s_bar[-1] - s_goal) 
        V_x[-1] = QN @ (s_bar[-1] - s_goal)
        V_xx[-1] = QN # final Hessian of cost-to-go
        

        # Q-values
        Q_k = np.zeros((N, 1)) 
        Q_x = np.zeros((N, n)) 
        Q_u = np.zeros((N, m)) 
        Q_xx = np.zeros((N,  n,n))
        Q_uu = np.zeros((N, m,m))
        Q_ux = np.zeros((N, m,n))

        # Backward pass
        for k in range(N-1, -1, -1):
            # Linear terms
            qk[k]  = 2 * Q.T @ (s_bar[k] - s_goal) 
            rk[k] = 2 * R.T @ u_bar[k]

            # Stage-wise costs
            c_k[k] = 1/2 * (s_bar[k] - s_goal).T @ Q @ (s_bar[k] - s_goal) + 1/2 * u_bar[k].T @ R @ u_bar[k] # cost function c_k
            c_x[k] = Q.T @ (s_bar[k] - s_goal) 
            c_u[k] = R.T @ u_bar[k]
            c_xx[k] = Q
            c_uu[k] = R
            c_ux[k] = np.zeros((m,n))

            # Q-value matrices
            Q_k[k] = c_k[k] + V[k + 1] 
            Q_x[k] = c_x[k] + A[k].T @ V_x[k+1]
            Q_u[k] = c_u[k] + B[k].T @ V_x[k+1]
            Q_xx[k] = c_xx[k] + A[k].T @ V_xx[k+1] @ A[k]
            Q_uu[k] = c_uu[k] + B[k].T @ V_xx[k+1] @ B[k]
            Q_ux[k] = c_ux[k] + B[k].T @ V_xx[k+1] @ A[k]

            # y[k] = -np.linalg.inv(Q_uu[k]) @ Q_u[k]
            # Y[k] = -np.linalg.inv(Q_uu[k]) @ Q_ux[k]
            y[k] = -np.linalg.solve(Q_uu[k], Q_u[k])
            Y[k] = -np.linalg.solve(Q_uu[k], Q_ux[k])

            # # value function updates
            V[k] = Q_k[k] - 1/2 * y[k].T @ Q_uu[k] @ y[k]
            V_x[k] = Q_x[k] - Y[k].T @ Q_uu[k] @ y[k]
            V_xx[k] = Q_xx[k] - Y[k].T @ Q_uu[k] @ Y[k]

            # value function from Yuval's paper, same amount of iterations!
            # V[k] = 1/2 * y[k].T @ Q_uu[k] @ y[k] + y[k].T @ Q_u[k]
            # V_x[k] = Q_x[k] + Y[k].T @ Q_uu[k] @ y[k] + Y[k].T @ Q_u[k] + Q_ux[k].T @ y[k]
            # V_xx[k] = Q_xx[k] + Y[k].T @ Q_uu[k] @ Y[k] + Y[k].T @ Q_ux[k] +Q_ux[k].T @ Y[k]
            
        # Forward pass
        s_bar_new = s_bar.copy()
        u_bar_new = u_bar.copy()
        # s_bar_new[0] = s0
        for k in range(N):
            ds[k] = s_bar_new[k] - s_bar[k]
            du[k] = y[k] + Y[k] @ ds[k]

            u_bar_new[k] = u_bar[k] + du[k]
            s_bar_new[k+1] = f(s_bar_new[k], u_bar_new[k])
            
        # print(f"max du: {np.max(np.abs(du))}")
        # print(f"max ds: {np.max(np.abs(ds))}")
        # print(f"max Y: {np.max(np.abs(Y))}")
        # print(f"max y: {np.max(np.abs(y))}")
        # print(f"max V: {np.max(np.abs(V))}")
        # print(f"max V_x: {np.max(np.abs(V_x))}")
        # print(f"max V_xx: {np.max(np.abs(V_xx))}")
        # print(f"min Q_uu: {np.min(np.abs(Q_uu))}")
        # # breakpoint()

        u_bar = u_bar_new.copy()
        s_bar = s_bar_new.copy()
            
        #######################################################################

        if np.max(np.abs(du)) < eps:
            print(f"wahoo! iLQR converged after {count} iterations")
            print(f"maximum state deviation: {np.max(np.abs(ds))}")
            print(f"maximum control deviation: {np.max(np.abs(du))}")
            print(f"gain Y : {Y[-1]}")
            print(f"offset y : {y[-1]}")
            converged = True
            break
    if not converged:
        raise RuntimeError("iLQR did not converge!")
    return s_bar, u_bar, Y, y


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.0  # pendulum mass
    mc = 10.0  # cart mass
    L = 1.0  # pendulum length
    g = 9.81  # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = jnp.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L),
        ]
    )
    return ds


# Define constants
n = 4  # state dimension
m = 1  # control dimension
Q = np.diag(np.array([10.0, 10.0, 2.0, 2.0]))  # state cost matrix
R = 1e-2 * np.eye(m)  # control cost matrix
QN = 1e2 * np.eye(n)  # terminal state cost matrix
s0 = np.array([0.0, 0.0, 0.0, 0.0])  # initial state
s_goal = np.array([0.0, np.pi, 0.0, 0.0])  # goal state
T = 10.0  # simulation time
dt = 0.1  # sampling time
animate = True  # flag for animation
closed_loop = False  # flag for closed-loop control

# Initialize continuous-time and discretized dynamics
f = jax.jit(cartpole)
fd = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

# Compute the iLQR solution with the discretized dynamics
print("Computing iLQR solution ... ", end="", flush=True)
start = time.time()
t = np.arange(0.0, T, dt)
N = t.size - 1
s_bar, u_bar, Y, y = ilqr(fd, s0, s_goal, N, Q, R, QN)
print("done! ({:.2f} s)".format(time.time() - start), flush=True)

# Simulate on the true continuous-time system
print("Simulating ... ", end="", flush=True)
start = time.time()
s = np.zeros((N + 1, n))
u = np.zeros((N, m))
s[0] = s0
for k in range(N):
    # PART (d) ################################################################
    # INSTRUCTIONS: Compute either the closed-loop or open-loop value of
    # `u[k]`, depending on the Boolean flag `closed_loop`.
    if closed_loop:
        u[k] = u_bar[k] + y[k] + Y[k] @ (s[k] - s_bar[k])
        # raise NotImplementedError()
    else:  # do open-loop control
        u[k] = u_bar[k]
        # raise NotImplementedError()
    ###########################################################################
    s[k + 1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k : k + 2])[1]
print("done! ({:.2f} s)".format(time.time() - start), flush=True)

# Plot
fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
plt.subplots_adjust(wspace=0.45)
labels_s = (r"$x(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{\theta}(t)$")
labels_u = (r"$u(t)$",)
for i in range(n):
    axes[i].plot(t, s[:, i])
    axes[i].set_xlabel(r"$t$")
    axes[i].set_ylabel(labels_s[i])
for i in range(m):
    axes[n + i].plot(t[:-1], u[:, i])
    axes[n + i].set_xlabel(r"$t$")
    axes[n + i].set_ylabel(labels_u[i])
if closed_loop:
    plt.savefig("cartpole_swingup_cl.png", bbox_inches="tight")
else:
    plt.savefig("cartpole_swingup_ol.png", bbox_inches="tight")
plt.show()

if animate:
    fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
    ani.save("cartpole_swingup.mp4", writer="ffmpeg")
    plt.show()

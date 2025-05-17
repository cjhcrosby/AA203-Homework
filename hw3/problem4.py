import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_ellipsoid_points(M, num_points=100):
    """Generate points on a 2-D ellipsoid.

    The ellipsoid is described by the equation
        '{ x | x.T @ inv(M) @ x <= 1 }',
    where 'inv(M)/ denotes the inverse of the matrix argument 'M'.

    The returned array has shape (num_points, 2) .
    """

    L = np.linalg.cholesky(M)
    theta = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    x = u @ L.T
    return x

def get_X(rx, num_points=100):
    """
    X = \set{x : norm(x)**2 <= rx^{2}}
    """
    # compute the ellipsoid bounding X
    X = np.array([[rx*np.cos(theta), rx*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, num_points)])
    return X

def get_AXT(X_T, A, num_points=100):
    """
    AXT = A @ X_T
    """
    # compute the ellipsoid bounding AXT
    AXT = np.array([[A[0,0]*x[0] + A[0,1]*x[1],     
                     A[1,0]*x[0] + A[1,1]*x[1]] for x in X_T])
    return AXT

def do_mpc(
    x0: np.ndarray, # initial state
    A: np.ndarray,  # system dynamics matrix
    B: np.ndarray,  # control input matrix
    P: np.ndarray,  # terminal cost matrix
    Q: np.ndarray,  # state cost matrix
    R: np.ndarray,  # control cost matrix
    N: int,         # prediction horizon
    rx: float,      # initial state inf-norm constraint
    ru: float,      # initial control inf-norm constraint
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve the MPC problem starting at state `x0`."""
    n, m = Q.shape[0], R.shape[0]
    x_cvx = cp.Variable((N + 1, n))
    u_cvx = cp.Variable((N, m))

    # PART (a): YOUR CODE BELOW ###############################################
    # INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.

    cost = 0.0 # initialize cost
    constraints = [] # initialize constraints list
    cost += cp.quad_form(x_cvx[-1], P) # terminal cost
    constraints.append(x_cvx[0] == x0) # initial condition constraint
    constraints.append(cp.norm(x_cvx[0]) <= rx) # intial state norm constraint
    constraints.append(cp.norm(u_cvx[0]) <= ru) # initial control norm constraint
    constraints.append(cp.norm(x_cvx[-1]) <= rx) # terminal state norm constraint
    for k in range(N): # do the sum term
        cost += cp.quad_form(x_cvx[k], Q) + cp.quad_form(u_cvx[k], R) # step-wise cost argument
        constraints.append(x_cvx[k + 1] == A @ x_cvx[k] + B @ u_cvx[k]) # dynamics constraint
        constraints.append(cp.norm(x_cvx[k]) <= rx) # state norm constraint
        constraints.append(cp.norm(u_cvx[k]) <= ru) # control norm constraint
        
    # END PART (a) ############################################################

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(cp.CLARABEL)
    x = x_cvx.value
    u = u_cvx.value
    status = prob.status

    return x, u, status

if __name__ == "__main__":
    # dynamics
    A = np.array([[0.9, 0.6], [0, 0.8]])
    B = np.array([[0],[1]])

    n = A.shape[0]
    m = B.shape[0]

    # constraints
    N = 4
    rx = 5
    ru = 1
    Q = np.eye(n)
    R = np.eye(m)

    # # # part d
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    M = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [M << rx**2 * np.eye(n)]
    constraints += [
        cp.bmat([[M, A@M],[(A@M).T, M]]) >> 0
    ]
    prob = cp.Problem(cp.Maximize(cp.log_det(M)),
                    constraints)
    prob.solve()
    M = M.value
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution M is")
    print(M)
    print("W = inv(M):")
    print(np.linalg.inv(M))

    num_points = 100
    X_T = generate_ellipsoid_points(M, num_points=num_points)
    X = get_X(rx, num_points= num_points)
    AX_T = get_AXT(X_T, A, num_points=num_points)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.plot(X[:,0], X[:,1], 'grey', label=r'$\mathcal{X}$')
    plt.fill(X[:,0], X[:,1], alpha=0.2, color='grey')

    plt.plot(X_T[:,0], X_T[:,1], 'r', label=r'$\mathcal{X}_T$')
    plt.fill(X_T[:,0], X_T[:,1], alpha=0.2, color='red')

    plt.plot(AX_T[:,0], AX_T[:,1], 'b', label=r'$\mathbf{A}\mathcal{X}_T$')
    plt.fill(AX_T[:,0], AX_T[:,1], alpha=0.2, color='blue')

    
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')


    # # # part e - MPC problem
    x0 = np.array([0, -4.5])
    T = 15
    P = scipy.linalg.solve_continuous_lyapunov(A, Q)
    print("P = ", P)

    # mpc loop
    x = np.copy(x0)
    x_mpc = np.zeros((T, N + 1, n))
    u_mpc = np.zeros((T, N, m))
    for t in range(T):
        x_mpc[t], u_mpc[t], status = do_mpc(x, A, B.T, P, Q, R, N, rx, ru)
        if status == "infeasible":
            x_mpc = x_mpc[:t]
            u_mpc = u_mpc[:t]
            break
        x = A @ x + B.T @ u_mpc[t, 0, :]
        if t == 0:
            ax.plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k", label="MPC trajectory")
        ax.plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k")

    # plot
    plt.plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], "-o", color="tab:blue", label="Trajectory")
    plt.legend(loc='upper right')
    plt.show()
    
    #plot control
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.plot(u_mpc[:, 0, 1], "-o", color='tab:orange')
    # breakpoint()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u_k$")
    ax.set_title("Control input")
    plt.show()
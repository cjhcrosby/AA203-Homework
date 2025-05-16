import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Constants
s_eye = np.array([15,15])
s_goal = np.array([19,9])
sigma = 10
n = 20
gamma = 0.95
m = 4
actions = [0,1,2,3] #["u", "d", "l", "r"]

def w(s):
    """Storm probability function"""
    return np.exp(-np.linalg.norm(s-s_eye)**2 / (2 * sigma**2))

def dynamics(s, a):
    s_next = s.copy()  
    
    # 0 (up)
    if a == 0:
        if s[1] < n-1: 
            s_next[1] += 1
    # 1 (down)
    elif a == 1:
        if s[1] > 0:  
            s_next[1] -= 1
    # 2 (left)
    elif a == 2:
        if s[0] > 0:  
            s_next[0] -= 1
    # 3 (right)
    elif a == 3:
        if s[0] < n-1:  
            s_next[0] += 1
            
    return s_next

def T(s):
    """
    Transition matrix T gives probabilities of next state given current state 
    """

    return (1-w(s))*np.eye(m) + w(s)/4 # transition probability matrix

def R(s):
    """Reward function
    """
    if np.array_equal(s, s_goal): 
         return 1.0
    else:
        return 0.0
        
def value_iteration(V, gamma, epsilon=1e-6):
    """
    Value iteration algorithm
    """
    count = 0
    pi = np.zeros((n, n), dtype=int) # policy

    V_new = V.copy()
    while True:
        delta = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                s = np.array([i, j])
                rvalue = np.zeros(len(actions)) # (R + gamma * V)
                for a_idx, a in enumerate(actions): # loop thru actions
                    s_next = dynamics(s,a_idx) # get next state given an action
                    rvalue[a_idx] = R(s_next) + gamma * V[s_next[0], s_next[1]] # stack values of each action fo 
                expected_returns = T(s) @ rvalue # T *  (R + gamma * V)

                V_new[i,j] = np.max(expected_returns) # max_a(T * (R + gamma * V))
                pi[i,j] = np.argmax(expected_returns) # argmax_a(T * (R + gamma * V))
                delta[i,j] = max(delta[i,j], abs(V_new[i, j] - V[i, j])) # convergence check per state
        count += 1
        # print("Iteration:", count)
        # print("Delta:", delta[i,j])
        if np.max(delta) < epsilon:
            print("Converged after", count, "iterations")
            break
        V = V_new.copy()
    return V_new, pi

def simulate_MDP(V, pi, s0, n_steps=100):
    """
    Simulate the MDP
    """
    s = s0.copy()
    trajectory = [s.copy()]
    for _ in range(n_steps):
        a_intended = pi[s[0], s[1]] # intended action
        # choose action based on intended action and transition probabilities
        a = random.choice(actions, p=T(s)[a_intended])
        s = dynamics(s, a)
        trajectory.append(s.copy())
        if np.array_equal(s, s_goal):
            break
    return trajectory

if __name__ == "__main__":
    V = np.zeros((n, n))
    s0 = np.array([0, 19])
    V, pi = value_iteration(V, gamma, epsilon=1e-6)
    trajectory = simulate_MDP(V, pi, s0=s0, n_steps=100)
    V = V.T
    pi = pi.T
    # plot value heat map
    plt.imshow(V, cmap=cm.hot, interpolation='nearest')
    plt.plot(s_goal[0], s_goal[1], 'go', markersize=10, label='Goal')
    plt.plot(s_eye[0], s_eye[1], 'ro', markersize=10, label='Storm')
    plt.legend()
    plt.colorbar()
    plt.title('Value Function Heat Map')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.gca().invert_yaxis() 
    plt.show()

    # plot policy heat map
    plt.imshow(pi, cmap=cm.plasma, interpolation='nearest')
    # assign values to arrows
    for i in range(n):
        for j in range(n):
            if pi[i, j] == 0: # up (inverted y-axis)
                plt.arrow(j, i, 0, 0.2, head_width=0.1, head_length=0.2, fc='red', ec='red')
            elif pi[i, j] == 1: # down (inverted y-axis)
                plt.arrow(j, i, 0, -0.2, head_width=0.1, head_length=0.2, fc='red', ec='red')
            elif pi[i, j] == 2: # left
                plt.arrow(j, i, -0.2, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
            elif pi[i, j] == 3: # right
                plt.arrow(j, i, 0.2, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.colorbar()
    plt.plot(s_goal[0], s_goal[1], 'go', markersize=10, label='Goal')
    plt.plot(s_eye[0], s_eye[1], 'ro', markersize=10, label='Storm')
    # plot trajectory
    plt.plot([s[0] for s in trajectory], [s[1] for s in trajectory], 'b-', label='Trajectory')
    plt.plot(trajectory[0][0], trajectory[0][1], 'bo', markersize=3, label=None)
    plt.plot(trajectory[-1][0], trajectory[-1][1], 'bx', markersize=3, label=None)
    plt.gca().invert_yaxis() 
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Policy Heat Map')
    plt.show()
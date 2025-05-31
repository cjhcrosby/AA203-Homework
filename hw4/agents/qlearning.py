from .Agent import Agent, Transition
from collections import deque
from typing import Union
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import math, torch, random
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
from typing import List



    
class QLearning(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24, use_gpu : bool=True) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, use_gpu)
        # # # # # set GPU to mps # # # # #
        if use_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using MPS GPU: {self.device}")
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"Using CPU: {self.device}")
        print(f"Using device: {self.device}")
        # # # # # # # # # # # # # # # # # #
        self.policy_network = self.build_network().to(self.device)
        self.buffer = deque([], maxlen=1000) # empty replay buffer with the 1000 most recent transitions
        self.agent_name = "qlearning"

        self.iteration = 0

    def eps_threshold(self) -> float: # epsilon threshold for e-greedy exploration
        eps_start, eps_end, eps_decay = 0.9, 0.05, 1000
        self.iteration += 1
        return eps_end + (eps_start - eps_end) * math.exp(-1 * self.iteration / eps_decay)

    def build_network(self) -> torch.nn.Module:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Construct and return a Multi-Layer Perceptron (MLP) representing the Q function. Recall that a Q function accepts
        ###     two arguments i.e., a state and action pair. For this implementation, your Q function will process an observation
        ###     of the state and produce an estimate of the expected, discounted reward for each available action as an output -- allowing you to 
        ###     select the prediction assosciated with either action.
        ###     2) Use a hidden layer dimension as specified by 'self.hidden_dim'.
        ###     3) Our solution implements a three layer MLP with ReLU activations on the first two hidden units.
        ###     But you are welcome to experiment with your own network definition!
        ###
        ### Please see the following docs for support:
        ###     nn.Sequential: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        ###     nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ###     nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.policy_network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),  # Input layer: state_dim to hidden_dim
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), # Hidden layer: hidden_dim to hidden_dim
            nn.ReLU(),                           
            nn.Linear(self.hidden_dim, self.action_dim)  # Output layer: hidden_dim to action_dim
        )
        return self.policy_network
    
        ###########################################################################
    
    def policy(self, state : Union[np.ndarray, torch.tensor], train : bool=False) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            # print("state shape:", state.shape)  
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) If train == True, sample from the policy with e-greedy exploration with a decaying epsilon threshold. We've already
        ###     implemented a function you can use to call the exploration threshold at any instance in the iteration i.e., 'self.eps_treshold()'.
        ###     2) If train == False, sample the action with the highest Q value as predicted by your network.
        ###     HINT: An exemplar implementation will need to use torch.no_grad() in the solution.
        ###
        ### Please see the following docs for support:
        ###     random.random: https://docs.python.org/3/library/random.html#random.random
        ###     torch.randint: https://docs.pytorch.org/docs/stable/generated/torch.randint.html
        ###     torch.argmax: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
        ###     torch.no_grad(): https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html

        with torch.no_grad():                       # Turn off gradient computation for Q-value inference
            q_values = self.policy_network(state)   # Get Q-values for the current state

        if train:   # Training using e-greedy
            if random.random() < self.eps_threshold():  # e-greedy
                return torch.randint(0, self.action_dim, (1,)).to(self.device) # Take the random action
            else:
                return torch.argmax(q_values, dim=1)  # return the action with the highest Q value
            
        else: # Not training, just return the action with the highest Q value
            return torch.argmax(q_values, dim=1) 
        ###########################################################################
    
    def sample_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self.buffer) > self.batch_size
        samples = random.sample(self.buffer, self.batch_size)

        states, actions, targets = [], [], []
        for i in range(self.batch_size):
            s, a, r, sp = samples[i].state.to(self.device), samples[i].action.item(), samples[i].reward.to(self.device), samples[i].next_state if samples[i].next_state is None else samples[i].next_state.to(self.device)
            states.append(s)
            actions.append(a)
            with torch.no_grad():
                targets.append(r if sp is None else r + self.gamma*torch.max(self.policy_network(sp)))
        try:
            return torch.cat(states), torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1), torch.cat(targets).unsqueeze(1)
        except RuntimeError as e:
            print(f"Error in sample_buffer: {e}")
            print(f"Buffer size: {len(self.buffer)}, Batch size: {self.batch_size}")
            raise e
    @staticmethod
    def plot_rewards(reward_history: List[int]) -> None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure()
        plt.plot(reward_history)
        # plot moving average reward
        window_size = 20
        moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(reward_history)), moving_avg, color='orange', label='Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Curve')
        filename = f"reward_curve_{current_time}.png"
        plt.savefig(filename)
        plt.show()
        print(f"Saved reward curve as {filename}")

    def train(self, env : gym.wrappers, num_episodes : int=200) -> None:
        # # debug device 
        # print(f"Training on device: {self.device}")
        # print(f"MPS available: {torch.backends.mps.is_available()}")
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implementing the training algorithm according to Algorithm 1 on page 5 in "Playing Atari with Deep Reinforcement Learning".
        ###     2) Importantly, only take a gradient step on your memory buffer if the buffer size exceeds the batch size hyperparameter. 
        ###     HINT: In our implementation, we used the AdamW optimizer.
        ###     HINT: Use the custom 'Transition' data structure to push observed (s, a, r, s') transitions to the memory buffer. Then, 
        ###     you can sample from the buffer simply by calling 'self.sample_buffer()'.
        ###     HINT: In our implementation, we clip the value of gradients to 100, which is optional.
        ###
        ### Please see the following docs for support:
        ###     torch.optim.AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        ###     torch.nn.MSELoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        ###     torch.nn.utils.clip_grad_value_: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        
        # Initialize the optimizer
        optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        ep_rewards = []
        for epoch in tqdm(range(num_episodes)):
            obs, info = env.reset() # Initialize the sequence
            terminated, truncated = False, False #Initialize the flags
            
            while not terminated and not truncated:
                a_tensor = self.policy(obs, train=True) # Sample action from e-greedy policy
                next_obs, r, terminated, truncated, info = env.step(a_tensor.item()) # Execute action in emulator and observe reward
                # Convert to tensors
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                r_tensor = torch.tensor([r], dtype=torch.float32).to(self.device)
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
                # Add transition to the buffer using the Transition class from Agent.py
                self.buffer.append(Transition(obs_tensor, a_tensor, r_tensor, next_obs_tensor if not terminated else None))
                obs = next_obs # Update the observation to the next one
                
                if len(self.buffer) > self.batch_size: # Gradient step after a batch is collected
                    states, actions, targets = self.sample_buffer() # Sample a batch from the buffer
                    states = states.reshape(-1, self.state_dim) # Ensure states are in the correct shape
                    # print(states.shape) # debug shape dim issue
                    optimizer.zero_grad()
                    q_values = self.policy_network(states).gather(1, actions) # Gather Q values for the actions taken
                    loss = criterion(q_values, targets)    
                    loss.backward() 
                    # print(f"Epoch {epoch}, Loss: {loss.item()}")
                    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100) # Optinally clip the gradients
                    optimizer.step()
        ###########################################################################

        
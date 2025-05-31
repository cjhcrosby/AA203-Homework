from .Agent import Agent
from typing import List
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch, datetime

class ReinforceAgent(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.agent_name = "reinforce"

    def build_network(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def policy(self, state: np.ndarray, train : bool=False) -> int:
        state = torch.tensor(state, dtype=torch.float)
        action_probs = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        if train:
            log_prob = action_distribution.log_prob(action)
            return action, log_prob
        else:
            return action

    def train(self, env: gym.wrappers, num_episodes: int=500) -> None:
        reward_history = []
        for episode in range(num_episodes):
            obs, info = env.reset(seed=1738)
            terminated, truncated = False, False

            log_probs = []
            rewards = []

            while not terminated and not truncated:
                action, log_prob = self.policy(obs, train=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                log_probs.append(log_prob)
                rewards.append(reward)

            self.learn(rewards, log_probs)
            total_reward = sum(rewards)
            reward_history.append(total_reward)
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

        self.plot_rewards(reward_history)

    def learn(self, rewards: list, log_probs: list) -> None:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implement the naive REINFORCE, REINFORCE with causality trick and REINFORCE with causality trick + a baseline to 'center' the returns.
        ###     2) After you've finished your implementation, please comment out all sections but the section you wish to evaluate for training.
        ###
        ### Please see the following docs for support:
        ###     torch.stack: https://docs.pytorch.org/docs/stable/generated/torch.stack.html
        𝛄 = 0.95  # discount factor
        
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # 1) Naive REINFORCE
        # total_return = sum(𝛄 ** t * rewards[t] for t in range(len(rewards)))
        # loss = -torch.sum(log_probs * total_return)
        # 2) REINFORCE with causality trick
        # returns = torch.zeros_like(rewards)
        # G = 0
        # for t in reversed(range(len(rewards))):
        #     G = rewards[t] + 𝛄 * G 
        #     returns[t] = G         
        # loss = -torch.sum(log_probs * returns)
        # 3) REINFORCE with causality trick and baseline to "center" the returns

        returns = torch.zeros_like(rewards)
        G = 0
        b = rewards.mean()
        for t in reversed(range(len(rewards))):
            G = rewards[t] + 𝛄 * G 
            returns[t] = G
        # b = returns.mean()
        # b = rewards.sum().mean()
        centered_returns = returns - b
        loss = -torch.sum(log_probs * centered_returns)   

        ###########################################################################

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)  # same gradient clipping as in Q-learning
        self.optimizer.step()
    
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
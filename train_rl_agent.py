# train_rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from traffic_env import TrafficIntersectionEnv

# 1. Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=2):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
                
    def __len__(self):
        return len(self.buffer)

# 3. Training Loop
def train_dqn():
    env = TrafficIntersectionEnv(max_steps=200)
    
    # Hyperparameters
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    lr = 0.001
    target_update_frequency = 10
    episodes = 250
    
    # Initialize networks
    policy_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=20000)
    
    epsilon = epsilon_start
    rewards_history = []
    
    print("Starting DQN Reinforcement Learning agent training...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = policy_net(state_t).argmax().item()
                    
            # Environment step
            next_state, reward, done, _, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Optimization step
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states_t = torch.tensor(states, dtype=torch.float32)
                actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32)
                dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                
                # Compute current Q values
                current_q = policy_net(states_t).gather(1, actions_t)
                
                # Compute target Q values
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + (1 - dones_t) * gamma * max_next_q
                    
                # Loss & backprop
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)
        
        # Update target network
        if episode % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if (episode + 1) % 25 == 0 or episode == 0:
            print(f"Episode {episode+1}/{episodes} | Avg Reward (last 10): {np.mean(rewards_history[-10:]):.1f} | Epsilon: {epsilon:.3f}")
            
    # Save trained policy net state dict
    torch.save(policy_net.state_dict(), "traffic_dqn.pt")
    print("DQN model trained successfully! Saved to traffic_dqn.pt")

if __name__ == "__main__":
    train_dqn()

"""
Reinforcement Learning Lab - Policy Gradient (REINFORCE)
Train an agent to solve CartPole using REINFORCE algorithm
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """
    Neural network that represents the policy π(a|s)
    Takes state as input and outputs probability distribution over actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass: returns log probabilities of actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.log_softmax(logits, dim=-1)
    
    def get_action(self, state):
        """
        Sample an action from the policy
        Returns: action, log_probability of that action
        """
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


class REINFORCEAgent:
    """
    REINFORCE Agent implementing vanilla policy gradient algorithm
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, device='cpu'):
        self.gamma = gamma  # Discount factor
        self.device = device
        
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for trajectories
        self.reset_trajectory()
        
    def reset_trajectory(self):
        """Reset storage for a new episode"""
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.get_action(state_tensor)
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward):
        """Store reward for the current step"""
        self.rewards.append(reward)
    
    def update_policy(self):
        """
        Update policy using REINFORCE algorithm
        Gradient ascent on expected return: ∇θ J(θ) = E[∇θ log π(a|s) * R]
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns (baseline reduction, reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)  # Negative because we're doing gradient ascent
        
        # Update policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        episode_reward = sum(self.rewards)
        self.reset_trajectory()
        
        return episode_reward


def train(env_name='CartPole-v1', num_episodes=1000, max_steps=500, 
          lr=1e-3, gamma=0.99, render=False, device='cpu', 
          eval_interval=50, target_score=475):
    """
    Train REINFORCE agent
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Max episode steps: {max_steps}")
    
    # Create agent
    agent = REINFORCEAgent(state_dim, action_dim, lr=lr, gamma=gamma, device=device)
    
    # Training statistics
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    print("\nStarting training...")
    print("=" * 60)
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward)
            episode_reward += reward
            
            state = next_state
            
            if done:
                break
        
        # Update policy after episode
        total_reward = agent.update_policy()
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        
        # Evaluation
        if episode % eval_interval == 0:
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Average reward (last 100): {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  Current episode reward: {total_reward:.2f}")
            
            # Check if solved
            if avg_reward >= target_score:
                print(f"\n🎉 Environment solved at episode {episode}!")
                print(f"   Average reward: {avg_reward:.2f} >= {target_score}")
                break
    
    env.close()
    
    # Plot training progress
    if len(episode_rewards) > 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
        if len(episode_rewards) >= 100:
            moving_avg = []
            for i in range(100, len(episode_rewards) + 1):
                moving_avg.append(np.mean(episode_rewards[i-100:i]))
            plt.plot(range(100, len(episode_rewards) + 1), moving_avg, 
                    color='red', linewidth=2, label='Moving Average (100)')
        plt.axhline(y=target_score, color='g', linestyle='--', label=f'Target: {target_score}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=target_score, color='r', linestyle='--', label=f'Target: {target_score}')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("\nTraining plot saved as 'training_progress.png'")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    eval_episodes = 10
    eval_rewards = []
    
    env = gym.make(env_name)
    
    for _ in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        agent.reset_trajectory()
    
    env.close()
    
    avg_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    print(f"Average reward over {eval_episodes} evaluation episodes: "
          f"{avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
    print(f"Max reward: {np.max(eval_rewards):.2f}")
    print(f"Min reward: {np.min(eval_rewards):.2f}")
    
    # Save model
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'final_reward': avg_eval_reward,
    }, 'reinforce_model.pth')
    print(f"\nModel saved as 'reinforce_model.pth'")
    
    return agent, episode_rewards


def test_model(model_path='reinforce_model.pth', env_name='CartPole-v1', 
               num_episodes=5, render=True):
    """
    Test a trained model
    """
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create environment
    env = gym.make(env_name, render_mode='human' if render else None)
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    
    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"Testing for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            log_probs = policy(state_tensor)
            action = torch.exp(log_probs).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='REINFORCE Policy Gradient Training')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--target_score', type=float, default=475.0,
                       help='Target average reward to consider solved')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='Evaluate every N episodes')
    parser.add_argument('--test', action='store_true',
                       help='Test a trained model instead of training')
    parser.add_argument('--model_path', type=str, default='reinforce_model.pth',
                       help='Path to model for testing')
    parser.add_argument('--test_episodes', type=int, default=5,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    if args.test:
        test_model(args.model_path, args.env, args.test_episodes, render=False)
    else:
        train(
            env_name=args.env,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            eval_interval=args.eval_interval,
            target_score=args.target_score
        )


if __name__ == '__main__':
    main()


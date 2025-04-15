# Step 1: Install Required Libraries
!pip install gym stable-baselines3 shimmy matplotlib imageio[pyav] moviepy seaborn

# Step 2: Import Libraries
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import os
from google.colab import files
from stable_baselines3 import PPO
import seaborn as sns

# Step 3: Define the Custom Environment (TreasureHuntEnv)
class TreasureHuntEnv(gym.Env):
    def __init__(self, grid_size=5, dynamic_difficulty=False):
        super(TreasureHuntEnv, self).__init__()
        
        # Grid size
        self.grid_size = grid_size
        
        # Dynamic difficulty scaling
        self.dynamic_difficulty = dynamic_difficulty
        self.difficulty_level = 1  # Initial difficulty level
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Invoke Special Action
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=int)
        
        # Initialize the grid
        self.reset()
    
    def reset(self):
        # Reset the environment
        self.agent_pos = [0, 0]
        
        # Adjust number of treasures and traps based on difficulty level
        num_treasures = 3 + self.difficulty_level
        num_traps = 2 + self.difficulty_level
        
        self.treasures = [(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) for _ in range(num_treasures)]
        self.traps = [(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) for _ in range(num_traps)]
        
        self.invocations_left = 1  # Reset invocations
        return self.agent_pos
    
    def step(self, action):
        # Move the agent based on the action
        reward = -1  # Default penalty for each step
        done = False
        
        if action == 4 and self.invocations_left > 0:  # Special Invocation (Teleport)
            self.agent_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            reward += 5  # Small reward for using invocation
            self.invocations_left -= 1
        else:
            if action == 0:  # Up
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:  # Down
                self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            elif action == 2:  # Left
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 3:  # Right
                self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        
        # Check for rewards/penalties
        if tuple(self.agent_pos) in self.treasures:
            reward += 10  # Reward for collecting a treasure
            self.treasures.remove(tuple(self.agent_pos))
        
        if tuple(self.agent_pos) in self.traps:
            reward -= 5  # Penalty for stepping on a trap
            done = True  # End the episode
        
        # End the episode if all treasures are collected
        if len(self.treasures) == 0:
            done = True
            self.difficulty_level += 1  # Increase difficulty for next episode (if dynamic difficulty is enabled)
        
        return self.agent_pos, reward, done, {}
    
    def render(self):
        # Render the environment (text-based for now)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Place treasures
        for t in self.treasures:
            grid[t[0]][t[1]] = 'T'
        
        # Place traps
        for tr in self.traps:
            grid[tr[0]][tr[1]] = 'X'
        
        # Place agent
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        # Print the grid
        for row in grid:
            print(' '.join(row))
        print(f"Invocations Left: {self.invocations_left}")

# Step 4: Train the Agent
env = TreasureHuntEnv(grid_size=5, dynamic_difficulty=True)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10_000)

# Step 5: Visualize and Save the Game as a Video
def visualize_and_save_game(env, model, num_steps=300, output_file="game_animation.mp4"):
    grid_size = env.grid_size
    frames = []
    rewards_over_time = []
    
    # Initialize the grid
    grid = np.full((grid_size, grid_size), '.', dtype=str)
    
    # Function to update the grid
    def update_grid(agent_pos, treasures, traps):
        grid.fill('.')
        for t in treasures:
            grid[t[0], t[1]] = 'T'  # Treasure
        for tr in traps:
            grid[tr[0], tr[1]] = 'X'  # Trap
        grid[agent_pos[0], agent_pos[1]] = 'A'  # Agent
    
    # Reset the environment
    obs = env.reset()
    
    for step in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards_over_time.append(reward)
        
        # Update the grid
        update_grid(obs, env.treasures, env.traps)
        
        # Render the grid as an image
        fig, ax = plt.subplots(figsize=(6, 6))  # Increased resolution
        ax.imshow([[1 if c != '.' else 0 for c in row] for row in grid], cmap='cool', interpolation='none')
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j, i, grid[i][j], ha='center', va='center', color='black', fontsize=12)  # Larger font
        ax.set_title(f"Step: {step + 1}", fontsize=14)
        
        # Save the figure as an image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to numpy array
        frames.append(image)
        plt.close(fig)  # Close the figure to free memory
        
        if done:
            obs = env.reset()
    
    # Save frames as a video with increased quality
    writer = imageio.get_writer(output_file, fps=10, quality=9)  # Increased FPS and quality
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"Animation saved as '{output_file}'")
    
    # Plot rewards over time
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_over_time, label="Reward per Step")
    plt.title("Reward Distribution Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("reward_plot.png")
    plt.show()
    
    return output_file

# Run the visualization and save the video
output_file = visualize_and_save_game(env, model)

# Step 6: Download the Video File and Reward Plot
files.download(output_file)
files.download("reward_plot.png")

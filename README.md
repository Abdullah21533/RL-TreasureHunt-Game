# 🧠 Reinforcement Learning: Treasure Hunt Game (RL-TreasureHunt-Game)

An innovative grid-based environment built using **Reinforcement Learning** to train an agent to collect treasures, avoid traps, and learn efficiently through adaptive difficulty and real-time visualizations.

> 🚀 Designed for research, learning, and showcasing RL capabilities in an interactive and engaging format.

---

## 🎮 Game Description

- **Environment**: 2D Grid World with randomly placed **treasures**, **traps**, and **walls**.
- **Agent**: Learns to navigate the world using PPO (Proximal Policy Optimization).
- **Objective**: Maximize rewards by collecting treasures and avoiding traps.
- **Challenges**:
  - Penalty for each step to encourage efficiency.
  - Rewards/Penalties:
    - `+10` for collecting treasure.
    - `-5` for hitting a trap.
    - `-1` per move.

---

## 🧠 Key Innovations

### 🔄 Dynamic Difficulty Scaling
The environment adjusts the number of traps and treasures based on the agent’s performance—creating a continuously challenging space to learn.

### 🤖 Multi-Agent Ready (Future Work)
Designed with architecture extensibility to support **multi-agent collaboration** or **competitive agents**.

### 🧮 Custom Reward Function
Incorporates:
- Step penalty for inefficiency.
- Bonus for intelligent teleportation (limited usage).
- Future support for proximity-based rewards and teamwork.

### 📊 Scientific Metrics
Tracks and visualizes:
- **Reward distribution**
- **Average steps per episode**
- **Success rate**
- **Exploration vs Exploitation trade-offs**

### 🔥 Visualization Enhancements
- Real-time animated game grid using Matplotlib
- Heatmaps of agent’s movement patterns (using Seaborn)
- Exported video (MP4) of the agent in action
- Reward learning graph (`reward_plot.png`)

---

## 📂 Project Structure


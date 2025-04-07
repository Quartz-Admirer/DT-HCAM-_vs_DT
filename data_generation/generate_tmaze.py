import numpy as np
from tqdm import trange
from envs.passive_tmaze_env import PassiveTMazeEnv

def optimal_policy(agent_pos, maze_center):
    if agent_pos[0] < maze_center[0]:
        return 1  
    elif agent_pos[0] > maze_center[0]:
        return 0  
    elif agent_pos[1] < maze_center[1]:
        return 3  
    elif agent_pos[1] > maze_center[1]:
        return 2  
    return 0 

def compute_returns(rewards, gamma=1.0):
    returns = np.zeros_like(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns

def generate_tmaze_data(num_episodes=500, max_steps=50, gamma=1.0, seed=42, save_path="data/passive_tmaze/optimal_trajectories.npz"):
    np.random.seed(seed)
    env = PassiveTMazeEnv()
    episodes_data = []
    maze_center = [env.maze_size // 2, env.maze_size // 2]

    for ep in trange(num_episodes, desc="Generating Optimal T-Maze data"):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        for step in range(max_steps):
            action = optimal_policy(obs, maze_center)

            states.append(np.array(obs, dtype=np.float32))
            actions.append(action)

            obs, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

            if done or truncated:
                break

        if len(states) == 0:
            continue

        returns = compute_returns(rewards, gamma=gamma)

        episodes_data.append((
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(returns, dtype=np.float32)
        ))

    n_success = sum(np.any(ep[2] > 0) for ep in episodes_data)
    print(f"\nУспешных эпизодов: {n_success}/{len(episodes_data)}")

    states_all = np.array([ep[0] for ep in episodes_data], dtype=object)
    actions_all = np.array([ep[1] for ep in episodes_data], dtype=object)
    returns_all = np.array([ep[2] for ep in episodes_data], dtype=object)

    print("states dtype:", states_all[0].dtype)
    print("states shape:", states_all[0].shape)

    np.savez_compressed(
        save_path,
        states=states_all,
        actions=actions_all,
        returns=returns_all
    )

    print(f"\nOptimal dataset saved to {save_path}")

if __name__ == "__main__":
    generate_tmaze_data()

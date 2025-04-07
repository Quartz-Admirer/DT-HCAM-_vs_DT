# data_generation/generate_minigrid_random.py

import argparse
import numpy as np
import gymnasium as gym
from tqdm import trange
import random
import os

def compute_returns(rewards, gamma=1.0):
    returns = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns

def process_obs_for_storage(obs):

    if not isinstance(obs, dict) or "image" not in obs or "direction" not in obs:
         
         raise ValueError(f"Неожиданный формат наблюдения MiniGrid (ожидался dict с 'image', 'direction'): {type(obs)}")

    img = obs["image"] 
    direction = obs["direction"]

    if not isinstance(img, np.ndarray): img = np.array(img)

    expected_view_size = 7 
    if len(img.shape) != 3 or img.shape[0] != img.shape[1] or img.shape[2] != 3:
        print(f"Warning: Ожидалась форма ЧАСТИЧНОГО вида (V, V, 3), получено {img.shape}. Используем как есть.")

    img_flat = img.flatten()
    state_vec = np.concatenate([img_flat, [direction]]).astype(np.float32)
    return state_vec

def generate_random_minigrid_data(
    env_id="MiniGrid-MemoryS17Random-v0",
    num_episodes=10000,
    max_steps=300,
    gamma=0.99,
    seed=42,
    save_dir="data/minigrid_memory"):

    """Генерирует датасет со случайной политикой для стандартной среды MiniGrid, используя стандартные наблюдения."""
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    env_short_name = env_id.split('-')[1].lower() if '-' in env_id else env_id.lower()
    save_path = os.path.join(save_dir, f"random_{env_short_name}_trajectories.npz")

    try:
        env = gym.make(env_id)
        print(f"Создана среда: {env_id}")

        action_dim = env.action_space.n
        obs_sample, _ = env.reset(seed=seed)
        state_vec_sample = process_obs_for_storage(obs_sample)
        state_dim = len(state_vec_sample)
        print(f"Пространство действий: {env.action_space}")
        print(f"Observation space sample (keys): {obs_sample.keys() if isinstance(obs_sample, dict) else 'Not a dict'}")
        print(f"Observation image shape (agent view): {obs_sample.get('image', np.array([])).shape}")
        print(f"Размерность состояния (state_dim): {state_dim}")
        print(f"Размерность действий (act_dim): {action_dim}")

    except Exception as e:
        print(f"Не удалось создать среду {env_id}: {e}")
        try:
             import minigrid.envs.memory
             print("Попытка импорта minigrid.envs.memory для регистрации...")
             env = gym.make(env_id)
        except Exception as e2:
             print(f"Повторная попытка создания среды также не удалась: {e2}")
             return

    episodes_data = []
    total_steps = 0
    non_zero_reward_episodes = 0

    for ep in trange(num_episodes, desc=f"Generating Random {env_id} data"):
        obs, _ = env.reset(seed=seed + ep)
        states_ep, actions_ep, rewards_ep = [], [], []
        episode_reward_sum = 0

        for step in range(max_steps):
            state_vec = process_obs_for_storage(obs)
            states_ep.append(state_vec)

            action = env.action_space.sample()
            actions_ep.append(action)

            obs, reward, done, truncated, info = env.step(action)
            rewards_ep.append(reward)
            episode_reward_sum += reward

            if done or truncated:
                break

        total_steps += len(states_ep)
        if episode_reward_sum > 0: non_zero_reward_episodes += 1

        if len(rewards_ep) > 0:
            returns_ep = compute_returns(rewards_ep, gamma=gamma)
            ep_tuple = (
                np.array(states_ep, dtype=np.float32),
                np.array(actions_ep, dtype=np.int32),
                np.array(returns_ep, dtype=np.float32)
            )
            episodes_data.append(ep_tuple)

    if not episodes_data: print("\nНе собрано ни одного эпизода!"); return

    print(f"\nСобрано эпизодов: {len(episodes_data)}")
    print(f"Эпизодов с наградой > 0: {non_zero_reward_episodes}")
    episode_returns = [ep[2][0] for ep in episodes_data if len(ep[2]) > 0]
    max_return = max(episode_returns) if episode_returns else 0
    avg_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0
    print(f"Макс. суммарная награда (RTG[0]): {max_return:.4f}")
    print(f"Сред. суммарная награда (RTG[0]): {avg_return:.4f}")

    states_all = np.array([ep[0] for ep in episodes_data], dtype=object)
    actions_all = np.array([ep[1] for ep in episodes_data], dtype=object)
    returns_all = np.array([ep[2] for ep in episodes_data], dtype=object)

    print(f"\nСохранение датасета в {save_path}...")
    metadata_dict=dict(
            env_id=env_id, state_dim=state_dim, action_dim=action_dim,
            num_episodes=len(episodes_data), data_type="random", gamma=gamma,
            max_steps_per_episode=max_steps, max_return=max_return,
            avg_return=avg_return, non_zero_reward_episodes=non_zero_reward_episodes
        )
    np.savez_compressed(
        save_path,
        states=states_all, actions=actions_all, returns=returns_all,
        metadata=metadata_dict
    )
    print(f"Случайный датасет сохранен.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-MemoryS17Random-v0", help="ID стандартной среды MiniGrid")
    parser.add_argument("--save_dir", type=str, default="data/minigrid_memory", help="Папка для сохранения")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Количество эпизодов")
    parser.add_argument("--max_steps", type=int, default=300, help="Макс. шагов в эпизоде")
    parser.add_argument("--gamma", type=float, default=0.99, help="Фактор дисконтирования")
    parser.add_argument("--seed", type=int, default=42, help="Сид")
    args = parser.parse_args()

    generate_random_minigrid_data(
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        seed=args.seed,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
# train/evaluation.py

import torch
import numpy as np
import math
import collections
import gymnasium as gym


def process_obs_minigrid(obs):
    if not isinstance(obs, dict) or "image" not in obs or "direction" not in obs:
        if isinstance(obs, np.ndarray):
             return obs.astype(np.float32)
        raise ValueError(f"Неожиданный формат наблюдения MiniGrid: {type(obs)}, ожидался dict с 'image', 'direction'.")

    img = obs["image"]
    direction = obs["direction"]

    if not isinstance(img, np.ndarray): img = np.array(img)
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Ожидалась форма изображения (H, W, 3), получено {img.shape}")

    img_flat = img.flatten()
    state_vec = np.concatenate([img_flat, [direction]]).astype(np.float32)
    return state_vec

def process_obs_tmaze(obs):
    """Обработка наблюдения для T-Maze (уже numpy вектор)."""
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    else:
        try:
            return np.array(obs).astype(np.float32)
        except Exception as e:
             raise TypeError(f"Неподдерживаемый тип наблюдения для T-Maze: {type(obs)}. Ошибка: {e}")


def evaluate_model(
    model,
    env,         
    env_name,     
    num_episodes=10,
    max_steps=200,
    device='cuda',
    target_return=None,
    eval_seq_len=50
    ):

    model.eval()
    model.to(device)
    total_rewards = []

    if 'tmaze' in env_name.lower():
        process_obs_fn = process_obs_tmaze
        default_target_return = 1.0
        print(f"Evaluation: Using T-Maze observation processing. Default Target RTG: {default_target_return}")
    elif 'minigrid' in env_name.lower():
        process_obs_fn = process_obs_minigrid
        default_target_return = 0.2
        print(f"Evaluation: Using MiniGrid observation processing. Default Target RTG: {default_target_return}")
    else:
        print(f"Warning: Unknown environment type '{env_name}'. Using identity observation processing.")
        process_obs_fn = lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) else np.array(x).astype(np.float32)
        default_target_return = 2.0

    eval_target_return = target_return if target_return is not None else default_target_return


    use_hcam = hasattr(model, 'blocks') and hasattr(model.blocks[0], 'hcam')
    chunk_size = None
    actual_seq_len = eval_seq_len

    if use_hcam:
        try:
            chunk_size = model.blocks[0].hcam.chunk_size
            actual_seq_len = math.ceil(eval_seq_len / chunk_size) * chunk_size
            print(f"Evaluation: HCAM detected (chunk_size={chunk_size}). Using sequence length: {actual_seq_len}")
        except (AttributeError, TypeError):
            print("Evaluation: Warning - Could not detect chunk_size from HCAM model. Assuming standard DT seq len.")
            use_hcam = False

    model_max_input_len = model.max_length
    if actual_seq_len > model_max_input_len:
         print(f"Warning: Evaluation sequence length ({actual_seq_len}) > model's max_length ({model_max_input_len}). Evaluation might fail or use truncated history via model's positional embeddings.")

    print(f"Evaluation: Using target_return={eval_target_return}, actual_seq_len={actual_seq_len}, max_steps={max_steps}")

    for i in range(num_episodes):
        obs, info = env.reset()
        obs = process_obs_fn(obs)
        episode_reward = 0.0
        done = False
        truncated = False

        states_hist = collections.deque(maxlen=actual_seq_len)
        actions_hist = collections.deque(maxlen=actual_seq_len)
        rtgs_hist = collections.deque(maxlen=actual_seq_len)

        states_hist.append(obs)
        actions_hist.append(-1)
        rtgs_hist.append(eval_target_return)

        for t in range(max_steps):
            current_states = list(states_hist)
            current_actions = list(actions_hist)
            current_rtgs = list(rtgs_hist)

            current_len = len(current_states)
            pad_len = actual_seq_len - current_len

            if pad_len > 0:
                pad_state = np.zeros_like(current_states[0])
                pad_action = -1
                pad_rtg = current_rtgs[0]

                states_input_np = ([pad_state] * pad_len) + current_states
                actions_input_np = ([pad_action] * pad_len) + current_actions
                rtgs_input_np = ([pad_rtg] * pad_len) + current_rtgs
            else:
                states_input_np = current_states
                actions_input_np = current_actions
                rtgs_input_np = current_rtgs

            if len(states_input_np) != actual_seq_len:
                 print(f"ERROR in eval loop: state input length mismatch! Expected {actual_seq_len}, got {len(states_input_np)}")
                 break

            states_tensor = torch.from_numpy(np.array(states_input_np)).float().unsqueeze(0).to(device)
            actions_tensor = torch.from_numpy(np.array(actions_input_np)).long().unsqueeze(0).to(device)
            rtg_tensor = torch.from_numpy(np.array(rtgs_input_np)).float().unsqueeze(0).unsqueeze(-1).to(device)

            try:
                with torch.no_grad():
                    action_preds = model(states_tensor, actions_tensor, rtg_tensor)
                    action_logits = action_preds[0, -1]
                    action = torch.argmax(action_logits).item()
            except Exception as e:
                print(f"\nERROR during model inference in evaluation (step {t}): {e}")
                print(f"Input shapes: S={states_tensor.shape}, A={actions_tensor.shape}, R={rtg_tensor.shape}")
                done = True
                reward = 0

            if not done:
                new_obs, reward, done, truncated, info = env.step(action)
                new_obs = process_obs_fn(new_obs)
                episode_reward += reward
                
                states_hist.append(new_obs)
                actions_hist.append(action)
                next_rtg = rtgs_hist[-1] - reward
                rtgs_hist.append(next_rtg)

            if done or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {i+1}/{num_episodes} finished with reward: {episode_reward:.4f} in {t+1} steps")

    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    return avg_reward
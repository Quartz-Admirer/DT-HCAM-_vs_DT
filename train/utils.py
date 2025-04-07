# train/utils.py

import numpy as np
import torch
from torch.utils.data import Dataset
import gymnasium as gym
import math
import os

try:
    from envs.passive_tmaze_env import PassiveTMazeEnv
except ImportError:
    print("ERROR: Could not import PassiveTMazeEnv from envs.passive_tmaze_env. Check file existence and path.")
    raise

try:
    import minigrid.envs

except ImportError:
    print("Info: 'minigrid' library not found or failed to import. Standard MiniGrid environments might not be available via gym.make().")


class TrajectoryDataset(Dataset):

    def __init__(self, states, actions, returns_to_go, seq_len=50, chunk_size=None):
        super().__init__()
        if not isinstance(states, (list, np.ndarray)) or not hasattr(states, '__getitem__') or len(states) == 0:
            if len(states) == 0:
                 print(f"Warning: Initializing TrajectoryDataset with 0 trajectories.")
                 self.states = []
                 self.actions = []
                 self.returns_to_go = []
                 self.seq_len = seq_len
                 self.chunk_size = chunk_size
                 self.target_len = seq_len
                 if self.chunk_size is not None and self.chunk_size > 0:
                      self.target_len = math.ceil(self.seq_len / self.chunk_size) * self.chunk_size
                 return
            else:
                 raise TypeError(f"Expected states to be a list or ndarray of trajectories, got {type(states)}")


        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go
        self.seq_len = seq_len
        self.chunk_size = chunk_size

        self.target_len = seq_len
        if self.chunk_size is not None and self.chunk_size > 0:
            self.target_len = math.ceil(self.seq_len / self.chunk_size) * self.chunk_size
            
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        try:
            s_np = self.states[idx].astype(np.float32)
            a_np = self.actions[idx].astype(np.int32)
            r_np = self.returns_to_go[idx].astype(np.float32)
        except IndexError:
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.states)}")
        except AttributeError:
             raise TypeError(f"Data at index {idx} is not a numpy array or cannot be cast (S type: {type(self.states[idx])}, A type: {type(self.actions[idx])}, R type: {type(self.returns_to_go[idx])}).")

        if s_np.shape[0] == 0 or a_np.shape[0] == 0 or r_np.shape[0] == 0:
             raise ValueError(f"Empty sequence encountered in episode at index {idx}. Please filter dataset.")

        s = torch.from_numpy(s_np)
        a = torch.from_numpy(a_np).long()
        r = torch.from_numpy(r_np)

        start_idx = max(0, s.shape[0] - self.seq_len)
        s = s[start_idx:]
        a = a[start_idx:]
        r = r[start_idx:]

        current_len = s.shape[0]
        if current_len < self.target_len:
            pad_size = self.target_len - current_len

            state_dim = s.shape[1] if s.ndim == 2 else 1
            if s.ndim == 1: s = s.unsqueeze(-1)
            pad_s = torch.zeros((pad_size, state_dim), dtype=s.dtype)
            s = torch.cat([pad_s, s], dim=0)

            pad_a = torch.full((pad_size,), fill_value=-1, dtype=a.dtype)
            a = torch.cat([pad_a, a], dim=0)

            pad_r = torch.zeros((pad_size,), dtype=r.dtype)
            r = torch.cat([pad_r, r], dim=0)

        if s.shape[0] != self.target_len or a.shape[0] != self.target_len or r.shape[0] != self.target_len:
             raise RuntimeError(
                f"Padding error in __getitem__ idx {idx}: expected len {self.target_len}, "
                f"got S:{s.shape[0]}, A:{a.shape[0]}, R:{r.shape[0]}. "
                f"Original len was {len(s_np)}, current_len before pad was {current_len}."
             )

        return s, a, r


def load_dataset(dataset_path, seq_len=10, chunk_size=None):
    """Loads trajectory data and initializes the Dataset object."""
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
         print(f"ERROR: Dataset file not found at specified path: {dataset_path}")
         raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")
    try:
        with np.load(dataset_path, allow_pickle=True) as data:
            print(f"Dataset keys found: {list(data.keys())}")
            required_keys = ['states', 'actions', 'returns']
            if not all(key in data for key in required_keys):
                 raise ValueError(f"Dataset file {dataset_path} missing required keys: {required_keys}")

            states = data['states']
            actions = data['actions']
            returns_to_go = data['returns']

            if len(states) == 0:
                 print(f"Warning: Loaded dataset from {dataset_path} contains 0 trajectories.")

            metadata = data['metadata'].item() if 'metadata' in data else {}
            if metadata:
                 print(f"Loaded metadata: {metadata}")
            else:
                 print("No metadata found in dataset.")

    except Exception as e:
        print(f"ERROR: Failed to load or process dataset from {dataset_path}: {e}")
        raise

    dataset = TrajectoryDataset(
        states,
        actions,
        returns_to_go,
        seq_len=seq_len,
        chunk_size=chunk_size
    )
    return dataset, metadata


def make_env(env_name, seed=None):
    print(f"Attempting to create environment: '{env_name}' with seed={seed}")
    if env_name == 'passive_tmaze':
        try:
            env = PassiveTMazeEnv()
            print(f"Created custom environment: PassiveTMazeEnv")
            return env
        except Exception as e:
             print(f"ERROR: Failed to create PassiveTMazeEnv: {e}")
             raise
    else:
        try:
            import gymnasium as gym
            try:
                import minigrid.envs
            except ImportError:
                print("Info: 'minigrid' library not found or failed to import.")
            env = gym.make(env_name)
            print(f"Successfully created environment using gym.make('{env_name}')")

            print(f"Created env. Observation space: {env.observation_space}")
            if isinstance(env.observation_space, gym.spaces.Dict):
                 print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

                 if 'image' in env.observation_space.spaces:
                      print(f"Observation 'image' space shape: {env.observation_space['image'].shape}")
                 else:
                      print("Warning: Observation space is Dict but lacks 'image' key. Check env configuration.")

            return env

        except gym.error.NameNotFound:
             print(f"ERROR: Environment ID '{env_name}' not found in Gymnasium registry.")
             print("Ensure the environment name in your config is correct and registered.")
             print("For MiniGrid, try importing 'minigrid.envs' or using 'minigrid.register_envs()'.")
             raise
        except Exception as e:
             print(f"ERROR: Failed to create environment '{env_name}' using gym.make: {e}")
             raise
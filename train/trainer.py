# train/trainer.py

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb

try:
    from train.utils import load_dataset, make_env
    from train.eval import evaluate_model
    from models.decision_transformer import DecisionTransformer
    from models.dt_hcam import DTHCAM
except ImportError as e:
    print(f"ERROR: Failed to import project modules in trainer.py: {e}")
    print("Please ensure all required files (utils.py, evaluation.py, models/*.py, envs/*.py) exist and are accessible.")
    exit(1)

def train_model(config_path: str):
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at: {config_path}")
        return 

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        required_keys = [
            'env_name', 'dataset_path', 'state_dim', 'act_dim', 'hidden_size',
            'n_layers', 'n_heads', 'max_length', 'learning_rate', 'batch_size',
            'epochs', 'device', 'use_hcam', 'wandb_project'
        ]
        if config.get('use_hcam', False):
             required_keys.extend(['chunk_size', 'num_memory_slots'])

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"ERROR: Missing required keys in config file '{config_path}': {missing_keys}")
            return
    except Exception as e:
        print(f"ERROR: Failed to load or parse config file {config_path}: {e}")
        return

    try:
        model_type = 'DTHCAM' if config.get('use_hcam', False) else 'DT'
        run_name = f"{config['env_name']}_{model_type}"
        if config.get('use_hcam', False):
             run_name += f"_c{config.get('chunk_size','NA')}_m{config.get('num_memory_slots','NA')}"

        wandb.init(
            project=config["wandb_project"],
            config=config,
            name=run_name,
            group=f"{config['env_name']}", 
            job_type="train",
            reinit=True
        )
        print(f"WandB run '{run_name}' initialized in project '{config['wandb_project']}'.")
        use_wandb = True
    except Exception as e:
        print(f"ERROR: Failed to initialize WandB: {e}. Training will proceed without WandB logging.")
        use_wandb = False

    try:
        if not os.path.exists(config['dataset_path']):
            print(f"ERROR: Dataset path specified in config does not exist: {config['dataset_path']}")
            return

        print(f"Loading dataset with base_seq_len={config.get('seq_len', 50)}")
        try:
            dataset, metadata = load_dataset(
                config['dataset_path'],
                seq_len=config.get('seq_len', 50),
                chunk_size=config.get('chunk_size') if config.get('use_hcam', False) else None
            )
            print(f"Loaded dataset with {len(dataset)} trajectories.")
            if metadata: print(f"Dataset metadata: {metadata}")
        except Exception as e:
             print(f"ERROR: Failed during dataset loading: {e}")
             return

        if len(dataset) == 0:
             print(f"ERROR: Loaded dataset from {config['dataset_path']} is empty.")
             return

        if metadata and 'state_dim' in metadata and metadata['state_dim'] != config['state_dim']:
             print(f"\nCRITICAL WARNING: Config state_dim ({config['state_dim']}) != Dataset metadata state_dim ({metadata['state_dim']})!")
             print(f"Update config state_dim to {metadata['state_dim']} to match the dataset.\n")

        if metadata and 'action_dim' in metadata and metadata['action_dim'] != config['act_dim']:
             print(f"\nCRITICAL WARNING: Config act_dim ({config['act_dim']}) != Dataset metadata action_dim ({metadata['action_dim']})!")
             print(f"Update config act_dim to {metadata['action_dim']} to match the dataset.\n")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=True
        )

        try:
            env_seed = config.get('seed', None)
            env = make_env(config['env_name'], seed=env_seed)
        except Exception as e:
             print(f"ERROR: Failed to create environment '{config['env_name']}': {e}")
             return

        try:
            if config.get('use_hcam', False):
                print("Creating DTHCAM model...")
                model = DTHCAM(
                    state_dim=config['state_dim'],
                    act_dim=config['act_dim'],
                    hidden_size=config['hidden_size'],
                    n_layers=config['n_layers'],
                    n_heads=config['n_heads'],
                    max_length=config['max_length'],
                    chunk_size=config['chunk_size'],
                    num_memory_slots=config['num_memory_slots'],
                )
            else:
                print("Creating DecisionTransformer model...")
                model = DecisionTransformer(
                    state_dim=config['state_dim'],
                    act_dim=config['act_dim'],
                    hidden_size=config['hidden_size'],
                    n_layers=config['n_layers'],
                    n_heads=config['n_heads'],
                    max_length=config['max_length']
                )
            model = model.to(config['device'])
            print(f"Using model: {type(model).__name__}")
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {num_params:,}")
            if use_wandb: wandb.summary['model_parameters'] = num_params

        except Exception as e:
            print(f"ERROR: Failed to create model: {e}")
            return
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        print(f"\nStarting training on device: {config['device']}")
        print(f"Action dimension (act_dim from config): {config['act_dim']}")
        print(f"Number of epochs: {config['epochs']}")
        print(f"Learning rate: {config['learning_rate']}")

        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            processed_batches = 0
            model.train()

            for batch_idx, batch in enumerate(dataloader):
                states, actions, returns_to_go = batch
                states = states.to(config['device'], non_blocking=True)
                actions = actions.to(config['device'], non_blocking=True)
                returns_to_go = returns_to_go.to(config['device'], non_blocking=True)

                if epoch == 0 and batch_idx == 0 and actions.dtype != torch.long:
                     print(f"Warning: Actions tensor dtype is {actions.dtype}, expected torch.long.")

                optimizer.zero_grad()
                try:
                     pred_actions = model(states, actions, returns_to_go)
                except Exception as model_e:
                     print(f"\nERROR during model forward pass (Epoch {epoch+1}, Batch {batch_idx+1}): {model_e}")
                     print(f"Input Shapes: S={states.shape}, A={actions.shape}, R={returns_to_go.shape}")
                     raise model_e

                B, T, model_output_act_dim = pred_actions.shape
                if model_output_act_dim != config['act_dim']:
                     print(f"\nCRITICAL ERROR: Model output act_dim ({model_output_act_dim}) != config act_dim ({config['act_dim']})!")
                     raise ValueError("Model output dimension mismatch.")

                pred_actions_flat = pred_actions.view(-1, config['act_dim'])
                actions_flat = actions.view(-1)

                try:
                     loss = loss_fn(pred_actions_flat, actions_flat)
                except Exception as loss_e:
                      print(f"\nERROR during loss calculation (Epoch {epoch+1}, Batch {batch_idx+1}): {loss_e}")
                      print(f"Shapes: Preds={pred_actions_flat.shape}, Target={actions_flat.shape}")
                      print(f"Target dtype: {actions_flat.dtype}")
                      print(f"Target unique values: {torch.unique(actions_flat)}")
                      raise loss_e

                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"Warning: Loss is NaN or Inf at epoch {epoch+1}, batch {batch_idx+1}. Skipping optimizer step.")
                     continue

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                processed_batches += 1

            if processed_batches > 0: epoch_loss /= processed_batches
            else: epoch_loss = 0.0; print(f"WARNING: No batches processed in epoch {epoch+1}.")

            log_data = {"train/loss": epoch_loss, "epoch": epoch + 1}
            epoch_summary = f"[Epoch {epoch+1}/{config['epochs']}] Avg Loss: {epoch_loss:.4f}"

            eval_freq = config.get('eval_freq', 5)
            if (epoch + 1) % eval_freq == 0 or epoch == config['epochs'] - 1:
                model.eval()
                print(f"\n--- Evaluating model after Epoch {epoch+1} ---")
                try:
                    avg_reward = evaluate_model(
                        model=model,
                        env=env,
                        env_name=config['env_name'],
                        num_episodes=config.get('eval_episodes', 10),
                        max_steps=config.get('eval_max_steps', 300),
                        device=config['device'],
                        target_return=config.get('eval_target_return', None),
                        eval_seq_len=config.get('seq_len', 50)
                    )
                    print(f"--- Evaluation complete ---")
                    epoch_summary += f", Avg Eval Reward: {avg_reward:.4f}"
                    log_data["eval/avg_reward"] = avg_reward
                except Exception as eval_e:
                     print(f"ERROR during evaluation after epoch {epoch+1}: {eval_e}")
                     log_data["eval/avg_reward"] = float('nan')


            print(epoch_summary)
            if use_wandb: wandb.log(log_data)

        print("\nTraining finished.")

    except Exception as train_err:
         print(f"\nAn error occurred during training/evaluation: {train_err}")
         import traceback
         traceback.print_exc()

    finally:
        if use_wandb:
             print("Finishing WandB run...")
             wandb.finish()
             print("WandB run finished.")
        print("Exiting training script.")

def main():
    parser = argparse.ArgumentParser(description='Train Decision Transformer models.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file for the training run.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file specified does not exist: {args.config}")
        exit(1)

    train_model(args.config)

if __name__ == "__main__":
    main()
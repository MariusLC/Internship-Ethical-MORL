from moral.ppo import *
from envs.gym_wrapper import *

import torch
from tqdm import tqdm
import wandb
import argparse


# folder to load config file
CONFIG_PATH = "moral/"
CONFIG_FILENAME = "config_MORAL.yaml"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':

    config_yaml = load_config(CONFIG_FILENAME)

    # Fetch ratio args
    # parser = argparse.ArgumentParser(description='Preference Lambda.')
    # parser.add_argument('--lambd', nargs='+', type=int, required=True)
    # parser.add_argument('--env', type=int)
    # args = parser.parse_args()

    if args.lambd == None:
        raise NotImplementedError

    if args.env == 2:
        envi = 'randomized_v2'
    if args.env == 3:
        envi = 'randomized_v3'
    else: # ( null ou env inconnu)
        print("args.env = ",args.env,". Environnment missing or unknown, executing randomized_v3")
        envi = 'randomized_v3'
        args.env = 3


    # Init WandB & Parameters
    wandb.init(project='PPO', config={
        'env_id': envi,
        #'env_steps': 9e6,
        'env_steps': 1000,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.05,
        'lambd': [int(i) for i in args.lambd],
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)

        if train_ready:
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    torch.save(ppo.state_dict(), 'saved_models/ppo_v'+args.env+'_' + str(config.lambd) + '.pt')

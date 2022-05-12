from moral.ppo import PPO, TrajectoryDataset, update_policy
from envs.gym_wrapper import *
from moral.airl import *

from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import numpy as np
import pickle
import wandb
import argparse

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Fetch ratio args
    parser = argparse.ArgumentParser(description='Preference Lambda.')
    parser.add_argument('--lambd', nargs='+', type=int, required=True)
    parser.add_argument('--env', type=int)
    args = parser.parse_args()

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

    # Load demonstrations
    #expert_trajectories = pickle.load(open('demonstrations/ppo_demos_v3_[0,1,0,1].pk', 'rb'))
    exp_traj_file_name = 'demonstrations/ppo_v'+str(args.env)+"_["
    for i in range(len(args.lambd)):
        exp_traj_file_name += str(args.lambd[i])+','
    exp_traj_file_name = exp_traj_file_name[:-1]
    exp_traj_file_name += '].pk'
    expert_trajectories = pickle.load(open(exp_traj_file_name, 'rb'))

    # Init WandB & Parameters
    wandb.init(project='AIRL', config={
        'env_id': envi,
        'env_steps': 6e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'entropy_reg': 0,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=5e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=5e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # Logging
    objective_logs = []

    for t in tqdm(range((int(config.env_steps/config.n_workers)))):

        # Act
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        # Log Objectives
        objective_logs.append(rewards)

        # Calculate (vectorized) AIRL reward
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(airl_state, airl_next_state, config.gamma, airl_action_prob)
        airl_rewards = list(airl_rewards.detach().cpu().numpy() * [0 if i else 1 for i in done])

        # Save Trajectory
        train_ready = dataset.write_tuple(states, actions, airl_rewards, done, log_probs)

        if train_ready:
            # Log Objectives
            objective_logs = np.array(objective_logs).sum(axis=0)
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            objective_logs = []

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=config.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories.copy(), ppo=ppo,
                                                              batch_size=config.batchsize_discriminator)

            # Log Loss Statsitics
            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    #vec_env.close()
    # SAVE THE DISCRIMINATOR FOR THE MORAL STEP
    torch.save(discriminator.state_dict(), 'saved_models/discriminator_v'+args.env+'_'+str(config.lambd)+'.pt')

    # SAVE THE GENERATOR FOR THE MORAL STEP ?
    torch.save(ppo.state_dict(), 'saved_models/generator_v'+args.env+'_'+str(config.lambd)+'.pt')

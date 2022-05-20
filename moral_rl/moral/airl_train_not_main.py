from moral.ppo import PPO, TrajectoryDataset, update_policy
from envs.gym_wrapper import *
from moral.airl import *
from utils.evaluate_ppo import *

from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import numpy as np
import pickle
import wandb
import argparse

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# OLD
# def airl_train_n_experts(nb_experts, env, env_rad, lambd_list, ppo_filenames, discriminator_filenames, demo_path, model_path, model_ext, demo_ext):
#     for i in range(nb_experts):
#         demos_filename = demo_path+ppo_filenames+env+lambd_list[i]+demo_ext
#         discrim_filename = model_path+discriminator_filenames+env+lambd_list[i]+model_ext
#         airl_train_1_expert(env_rad+env, demos_filename, discrim_filename)

# NEW
def airl_train_n_experts(env, env_steps_airl, demos_filename, generators_filenames, discriminators_filenames):
    for i in range(len(generators_filenames)):
        airl_train_1_expert(env, env_steps_airl, demos_filename[i], generators_filenames[i], discriminators_filenames[i])



def airl_train_1_expert(env_id, env_steps_airl, demos_filename, generator_filename, discriminator_filename, prints=False):

    # Load demonstrations
    expert_trajectories = pickle.load(open(demos_filename, 'rb'))

    # Init WandB & Parameters
    wandb.init(
        project='AIRL',
        config={
            'env_id': env_id,
            #'env_steps': 6e6,
            'env_steps': env_steps_airl,
            'batchsize_discriminator': 512,
            'batchsize_ppo': 12,
            'n_workers': 12,
            'entropy_reg': 0,
            'gamma': 0.999,
            'epsilon': 0.1,
            'ppo_epochs': 5
            }, 
        reinit=True)
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
    discriminator = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
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
        # airl_rewards = discriminator.predict_reward_2(airl_state, airl_next_state, config.gamma, airl_action_prob)
        airl_advantages, airl_rewards = discriminator.predict_reward_2(airl_state, airl_next_state, config.gamma, airl_action_prob)
        airl_rewards = list(airl_rewards.detach().cpu().numpy() * [0 if i else 1 for i in done])
        airl_advantages = list(airl_advantages.detach().cpu().numpy() * [0 if i else 1 for i in done])

        # Save Trajectory
        # train_ready = dataset.write_tuple(states, actions, airl_rewards, done, log_probs)
        train_ready = dataset.write_tuple_2(states, actions, airl_rewards, airl_advantages, done, log_probs)

        if train_ready:
            # Log Objectives
            objective_logs = np.array(objective_logs).sum(axis=0)
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            objective_logs = []


            ####### TEST IF wandb is the one initilized in ppo_train_not_main or the one in this file
            # print(config.lambd)
            #######################

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=config.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories.copy(), ppo=ppo,
                                                              batch_size=config.batchsize_discriminator)

            if (prints):
                print('Discriminator Loss ', d_loss)
                print('Fake Accuracy ', fake_acc)
                print('Real Accuracy ', real_acc)

                print("mean discrim rew = ", sum(dataset.log_returns()))
                print("mean discrim adv = ", sum(dataset.log_advantages()))

                mean_ppo, std_ppo = evaluate_ppo(ppo, config)
                print("Mean returns per traj : ", mean_ppo)
                print("Std returns per traj : ", std_ppo)

                # mean_ppo, std_ppo, mean_discrim, std_discrim = evaluate_ppo_discrim(ppo, discriminator, config)
                # print("Mean rewards per episode : ", mean_ppo)
                # print("Std rewards per episode : ", std_ppo)
                # print("Mean discrim eval per episode : ", mean_discrim)
                # print("Std discrim eval per episode : ", std_discrim)

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
        torch.save(discriminator.state_dict(), discriminator_filename)

        # SAVE THE GENERATOR FOR THE MORAL STEP ?
        torch.save(ppo.state_dict(), generator_filename)
    

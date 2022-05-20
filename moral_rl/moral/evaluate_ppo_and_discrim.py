import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from envs.gym_wrapper import *

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import yaml
import os

import numpy as np
import math
import scipy.stats as st

from moral.active_learning import *
from moral.preference_giver import *
from moral.TEST_moral_train_not_main import *
from utils.generate_demos_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_EVALUATE.yaml"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def training_sampler(expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior=None):
    states = []
    action_probabilities = []
    next_states = []
    labels = []
    latents = []
    for i in range(batch_size):
        # 1 if (s,a,s') comes from expert, 0 otherwise
        # expert_boolean = np.random.randint(2)
        expert_boolean = 1 if i < batch_size/2 else 0
        if expert_boolean == 1:
            selected_trajectories = expert_trajectories
        else:
            selected_trajectories = policy_trajectories

        random_tau_idx = np.random.randint(len(selected_trajectories))
        random_tau = selected_trajectories[random_tau_idx]['states']
        random_state_idx = np.random.randint(len(random_tau)-1)
        state = random_tau[random_state_idx]
        next_state = random_tau[random_state_idx+1]

        # Sample random latent to condition ppo on for expert samples
        if latent_posterior is not None:
            if expert_boolean == 1:
                latent = latent_posterior.sample_prior()
                latent = latent.to(device)
            else:
                latent = torch.tensor(selected_trajectories[random_tau_idx]['latents']).to(device)

            action_probability, _ = ppo.forward(torch.tensor(state).float().to(device), latent)
            action_probability = action_probability.squeeze(0)
            latents.append(latent.cpu().item())
        else:
            action_probability, _ = ppo.forward(torch.tensor(state).float().to(device))
            action_probability = action_probability.squeeze(0)
        # Get the action that was actually selected in the trajectory
        selected_action = selected_trajectories[random_tau_idx]['actions'][random_state_idx]

        states.append(state)
        next_states.append(next_state)
        action_probabilities.append(action_probability[selected_action].item())
        labels.append(expert_boolean)

    return torch.tensor(states).float().to(device), torch.tensor(next_states).float().to(device), \
           torch.tensor(action_probabilities).float().to(device),\
           torch.tensor(labels).long().to(device), torch.tensor(latents).float().to(device)

def evaluate_discriminator(discriminator, optimizer, gamma, expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior=None):
    criterion = nn.CrossEntropyLoss()
    states, next_states, action_probabilities, labels, latents\
        = training_sampler(expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior)
    if len(latents) > 0:
        advantages = discriminator.forward(states, next_states, gamma, latents)
    else:
        advantages = discriminator.forward(states, next_states, gamma) 
    # Cat advantages and log_probs to (batch_size, 2)
    class_predictions = torch.cat([torch.log(action_probabilities).unsqueeze(1), advantages], dim=1)
    loss = criterion(class_predictions, labels)
    # Compute Accuracies
    label_predictions = torch.argmax(class_predictions, dim=1)
    predicted_fake = (label_predictions[labels == 0] == 0).float()
    predicted_expert = (label_predictions[labels == 1] == 1).float()

    loss.backward()
    return loss.item(), torch.mean(predicted_fake).item(), torch.mean(predicted_expert).item()


if __name__ == '__main__':

	config_yaml = load_config(CONFIG_FILENAME)

	#PARAMS CONFIG
	nb_experts = config_yaml["nb_experts"]
	nb_demos = config_yaml["nb_demos"]
	lambd_list = config_yaml["experts_weights"]

    # PATHS & NAMES
	vanilla = config_yaml["vanilla"]
	data_path = config_yaml["data_path"]
	model_name = config_yaml["model_name"]
	env_rad = config_yaml["env_rad"]
	env = config_yaml["env"]
	disc_path = config_yaml["disc_path"]
	expe_path = config_yaml["expe_path"]
	gene_path = config_yaml["gene_path"]
	demo_path = config_yaml["demo_path"]
	model_ext = config_yaml["model_ext"]
	demo_ext = config_yaml["demo_ext"]
	env_steps = config_yaml["env_steps"]
	rand = config_yaml["rand"]

	



    # Init WandB & Parameters
	wandb.init(project='test_discrim', config={
		'env_id': env_rad+env,
		'gamma': 0.999,
		'batchsize_discriminator': 512,
		#'env_steps': 9e6,
		# 'env_steps': 1000,
		# 'batchsize_ppo': 12,
		# 'n_workers': 12,
		# 'lr_ppo': 3e-4,
		# 'entropy_reg': 0.05,
		# 'lambd': [int(i) for i in args.lambd],
		# 'epsilon': 0.1,
		# 'ppo_epochs': 5
		})
	config = wandb.config

	# Create Environment
	vec_env = VecEnv(env_rad+env, 12)
	states = vec_env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = vec_env.action_space.n
	obs_shape = vec_env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]

	vanilla_path = ""
	if vanilla:
		vanilla_path += "Peschl/"

	experts_filenames = []
	demos_filenames = []
	# generators_filenames = []
	discriminators_filenames = []
	rand_filenames = []
	ppo_list = []
	rand_list = []
	discrim_list = []
	for i in range(nb_experts):
		expert_filename = data_path+expe_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext
		demos_filename = data_path+demo_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+demo_ext
		rand_filename = data_path+demo_path+vanilla_path+model_name+rand+env+"_"+str(lambd_list[i])+demo_ext
		generator_filename = data_path+gene_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext
		discriminator_filename = data_path+disc_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext

		# discriminator
		discrim_list = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
		discrim_list.load_state_dict(torch.load(discriminator_filename, map_location=torch.device('cpu')))
		optimizer_discriminator = torch.optim.Adam(discrim_list.parameters(), lr=5e-5)

		# experts
		ppo_list = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
		ppo_list.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

		# rand agents
		rand_list = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)

		if config_yaml["generate_demos"] :
			generate_demos_1_expert(nb_demos, env_rad+env, expert_filename, demos_filename)
			generate_demos_1_expert(nb_demos, env_rad+env, expert_filename, rand_filename)

		# Load demonstrations & evaluate ppo
		expert_trajectories = pickle.load(open(demos_filename, 'rb'))
		print(evaluate_ppo(ppo_list, config))
		rand_trajectories = pickle.load(open(rand_filename, 'rb'))
		print(evaluate_ppo(rand_list, config))
		d_loss, fake_acc, real_acc = evaluate_discriminator(discriminator=discrim_list,
															optimizer=optimizer_discriminator,
															gamma=config.gamma,
															expert_trajectories=expert_trajectories,
															policy_trajectories=rand_trajectories,
															ppo=rand_list,
															batch_size=config.batchsize_discriminator)

		print('Discriminator Loss ', d_loss)
		print('Fake Accuracy ', fake_acc)
		print('Real Accuracy ', real_acc)

	    # experts_filenames.append(data_path+ppo+env+lambd_list[i]+model_ext)
	    # demos_filenames.append("saved_models/demonstrations_"+env+lambd_list[i]+model_ext)
	    # rand_filenames.append("saved_models/demonstrations_rand_"+env+lambd_list[i]+model_ext)
	    # discriminators_filenames.append(data_path+discrim+env+lambd_list[i]+model_ext)
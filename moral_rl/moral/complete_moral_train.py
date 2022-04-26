from moral.ppo import *
from envs.gym_wrapper import *
from moral.ppo_train_not_main import ppo_train_n_experts
from utils.generate_demos_not_main import generate_demos_n_experts
from moral.airl_train_not_main import airl_train_n_experts
from moral.moral_train_not_main import moral_train_n_experts

import torch
from tqdm import tqdm
import wandb
import argparse
import os
import yaml


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

    #PARAMS CONFIG
    nb_experts = config_yaml["nb_experts"]
    ratio = config_yaml["ratio"]
    lambd_list = config_yaml["experts_weights"]
    nb_demos = config_yaml["nb_demos"]

    # PATHS & NAMES
    model_path = config_yaml["model_path"]
    demo_path = config_yaml["demo_path"]
    model_ext = config_yaml["model_ext"]
    demo_ext = config_yaml["demo_ext"]
    env = config_yaml["env"]
    env_rad = config_yaml["env_rad"]
    ppo_filenames = config_yaml["ppo_filenames"]
    discriminator_filenames = config_yaml["discriminator_filenames"]
    lambd_str_list = ["_"+str(w) for w in config_yaml["experts_weights"]]
    






    # TRAINING PPO AGENTS
    ppo_train_n_experts(nb_experts, env, env_rad, lambd_list, lambd_str_list, ppo_filenames, model_path, model_ext)

    # GENERATING DEMONSTRATIONS FROM EXPERTS
    generate_demos_n_experts(nb_experts, nb_demos, env, env_rad, lambd_str_list, ppo_filenames, demo_path, model_path, model_ext, demo_ext)

    # ESTIMATING EXPERTS REWARD FUNCTIONS THROUGH AIRL BASED ON THEIR DEMONSTRATIONS
    airl_train_n_experts(nb_experts, env, env_rad, lambd_str_list, ppo_filenames, discriminator_filenames, demo_path, model_path, model_ext, demo_ext)

    # ESTIMATING MORL EXPERT'S WEIGTHS THROUGH MORAL
    moral_train_n_experts(ratio, nb_experts, env, env_rad, lambd_str_list, ppo_filenames, discriminator_filenames, model_path, model_ext)
    # we shouldn't give acces to ppo_filename but demos instead ...
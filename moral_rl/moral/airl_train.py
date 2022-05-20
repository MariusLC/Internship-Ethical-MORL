from moral.ppo import PPO, TrajectoryDataset, update_policy
from envs.gym_wrapper import *
from moral.airl import *
from moral.airl_train_not_main import *

from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import numpy as np
import pickle
import wandb
import argparse
import os
import yaml

# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_AIRL.yaml"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    config_yaml = load_config(CONFIG_FILENAME)

    #PARAMS CONFIG
    nb_experts = config_yaml["nb_experts"]
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

    vanilla_path = ""
    if vanilla:
        vanilla_path += "Peschl/"

    envi = env_rad+env

    for i in range(nb_experts):
        expert_filename = data_path+expe_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext
        demos_filename = data_path+demo_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+demo_ext
        generator_filename = data_path+gene_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext
        discriminator_filename = data_path+disc_path+vanilla_path+model_name+env+"_"+str(lambd_list[i])+model_ext
        print("demos_filename = ", demos_filename)
        airl_train_1_expert(envi, env_steps, demos_filename, generator_filename, discriminator_filename, prints=True)



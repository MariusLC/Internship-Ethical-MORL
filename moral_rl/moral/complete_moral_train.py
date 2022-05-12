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
    data_path = config_yaml["data_path"]
    expe_path = config_yaml["expe_path"]
    demo_path = config_yaml["demo_path"]
    disc_path = config_yaml["disc_path"]
    gene_path = config_yaml["gene_path"]
    moral_path = config_yaml["moral_path"]
    model_ext = config_yaml["model_ext"]
    demo_ext = config_yaml["demo_ext"]
    env_rad = config_yaml["env_rad"]
    env = config_yaml["env"]
    model_name = config_yaml["model_name"]

    # OLD
    # model_path = config_yaml["model_path"]
    # demo_path = config_yaml["demo_path"]
    # model_ext = config_yaml["model_ext"]
    # demo_ext = config_yaml["demo_ext"]
    # env = config_yaml["env"]
    # env_rad = config_yaml["env_rad"]
    # ppo_filenames = config_yaml["ppo_filenames"]
    # discriminator_filenames = config_yaml["discriminator_filenames"]
    # generator_filenames = config_yaml["generator_filenames"]

    # lambd_str_list = ["_"+str(w) for w in config_yaml["experts_weights"]]

    experts_filenames = []
    demos_filenames = []
    generators_filenames = []
    discriminators_filenames = []
    moral_filename = data_path+moral_path+model_name+env+"_"+str(lambd_list)+model_ext
    for i in range(nb_experts):
        experts_filenames.append(data_path+expe_path+model_name+env+"_"+str(lambd_list[i])+model_ext)
        demos_filenames.append(data_path+demo_path+model_name+env+"_"+str(lambd_list[i])+demo_ext)
        generators_filenames.append(data_path+gene_path+model_name+env+"_"+str(lambd_list[i])+model_ext)
        discriminators_filenames.append(data_path+disc_path+model_name+env+"_"+str(lambd_list[i])+model_ext)


    # TRAINING PPO AGENTS
    ppo_train_n_experts(env_rad+env, lambd_list, experts_filenames)

    # GENERATING DEMONSTRATIONS FROM EXPERTS
    generate_demos_n_experts(nb_demos, env_rad+env, experts_filenames, demos_filenames)

    # ESTIMATING EXPERTS REWARD FUNCTIONS THROUGH AIRL BASED ON THEIR DEMONSTRATIONS
    airl_train_n_experts(env_rad+env, demos_filenames, generators_filenames, discriminators_filenames)

    # ESTIMATING MORL EXPERT'S WEIGTHS THROUGH MORAL
    moral_train_n_experts(ratio, env_rad+env, generators_filenames, discriminators_filenames, moral_filename)
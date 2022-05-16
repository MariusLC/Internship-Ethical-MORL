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

# folder to load config file
CONFIG_PATH = "moral/"
CONFIG_FILENAME = "config_MORAL.yaml"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

if __name__ == '__main__':


	# loss = nn.CrossEntropyLoss()

	# # Example of target with class indices
	# input = torch.randn(3, 5, requires_grad=True)
	# print(input)
	# target = torch.empty(3, dtype=torch.long).random_(5)
	# print(target)
	# output = loss(input, target)
	# print(output)
	# output.backward()
	# print(input.grad)

	# Example of target with class probabilities
	# input = torch.randn(3, 5, requires_grad=True)
	# print(input)
	# target = torch.randn(3, 5).softmax(dim=1)
	# print(target)
	# output = loss(input, target)
	# print(output)
	# output.backward()
	# print(input.grad)

	ratio = [1, 1, 3]
	# res_a = [10, 10, 30]
	res_a = [15, 15, 20]
	# res_b = [1, 1, 48]
	res_b = [48, 1, 1]

	ratio_normalized = [r/sum(ratio) for r in ratio]
	print(ratio_normalized)

	ret_a_copy = res_a.copy()
	ret_b_copy = res_b.copy()

	ret_a_normalized = []
	ret_b_normalized = []

	for i in range(len(ret_a_copy)):
		# To avoid numerical instabilities in KL
		ret_a_copy[i] += 1e-5
		ret_b_copy[i] += 1e-5

	ret_a_sum = sum(ret_a_copy)
	ret_b_sum = sum(ret_b_copy)

	for i in range(len(ret_a_copy)):
		ret_a_normalized.append(ret_a_copy[i]/ret_a_sum)
		ret_b_normalized.append(ret_b_copy[i]/ret_b_sum)

	# scipy.stats.entropy(pk, qk=None, base=None, axis=0) = S = sum(pk * log(pk / qk), axis=axis)
	kl_a = st.entropy(ret_a_normalized, ratio_normalized)
	kl_b = st.entropy(ret_b_normalized, ratio_normalized)
	print(kl_a)
	print(kl_b)

	# kl_a = st.entropy([1/2, 1/2], qk=[9/10, 1/10])
	# print(kl_a)
	# kl_a = st.entropy([9/10, 1/10], qk=[1/2, 1/2])
	# print(kl_a)
	# kl_a = st.entropy([1/2, 1/2], qk=[1/4, 3/4])
	# print(kl_a)
	# kl_a = st.entropy([9/10, 1/10], qk=[8/10, 2/10])
	# print(kl_a)
	# kl_a = st.entropy([9/10, 1/10], [8/10, 2/10])
	# print(kl_a)

	config_yaml = load_config(CONFIG_FILENAME)

	#PARAMS CONFIG
	nb_experts = 2
	lambd_list = [[0,0,1,1],[0,1,0,1]]
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

	experts_filenames = []
	demos_filenames = []
	generators_filenames = []
	discriminators_filenames = []
	moral_filename = data_path+moral_path+model_name+env+"_"+str(lambd_list)+model_ext
	for i in range(nb_experts):
	    experts_filenames.append("saved_models/Peschl_res/"+model_name+env+"_"+str(lambd_list[i])+model_ext)
	    discriminators_filenames.append("saved_models/Peschl_res/discriminator_"+env+"_"+str(lambd_list[i])+model_ext)

	moral_train_n_experts(ratio, env_rad+env, experts_filenames, discriminators_filenames, moral_filename)
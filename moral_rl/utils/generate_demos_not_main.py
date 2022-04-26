from tqdm import tqdm
from moral.ppo import PPO
import torch
from envs.gym_wrapper import GymWrapper
from envs.randomized_v2 import MAX_STEPS as max_steps_v2
from envs.randomized_v3 import MAX_STEPS as max_steps_v3
import pickle
import argparse

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_demos_n_experts(nb_experts, nb_demos, env, env_rad, lambd_list, ppo_filenames, demo_path, model_path, model_ext, demo_ext):
    for i in range(nb_experts):
        demos_filename = demo_path+ppo_filenames+env+lambd_list[i]+demo_ext
        ppo_filename = model_path+ppo_filenames+env+lambd_list[i]+model_ext
        ppo_train_1_expert(nb_demos, env_rad+env, demos_filename, ppo_filename)



def generate_demos_1_expert(nb_demos, env_id, demos_filename, ppo_filename):

    # Initialize Environment
    env = GymWrapper(env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    dataset = []
    episode = {'states': [], 'actions': []}
    episode_cnt = 0

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Load Pretrained PPO
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    ppo.load_state_dict(torch.load(ppo_filename, map_location=torch.device('cpu')))


    for t in tqdm(range((max_steps-1)*nb_demos)):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        episode['states'].append(states)
        # Note: Actions currently append as arrays and not integers!
        episode['actions'].append(actions)

        if done:
            next_states = env.reset()
            dataset.append(episode)
            episode = {'states': [], 'actions': []}

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    pickle.dump(dataset, open(demos_filename, 'wb'))

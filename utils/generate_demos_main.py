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

if __name__ == '__main__':

    # Fetch ratio args
    parser = argparse.ArgumentParser(description='Preference Lambda.')
    parser.add_argument('--lambd', nargs='+', type=int, required=True)
    parser.add_argument('--env', type=int)
    parser.add_argument('--nbdemos', type=int)
    args = parser.parse_args()

    if args.lambd == None:
        raise NotImplementedError
    if args.nbdemos == None: # a changer
        args.nbdemos = 1000
    n_demos = args.nbdemos

    max_steps = 0
    if args.env == 2:
        env_id = 'randomized_v2'
        max_steps = max_steps_v2
    if args.env == 3:
        env_id = 'randomized_v3'
        max_steps = max_steps_v3
    else: # ( null ou env inconnu)
        print("args.env = ",args.env,". Environnment missing or unknown, executing randomized_v3")
        env_id = 'randomized_v3'
        args.env = 3
        max_steps = max_steps_v3




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
    ppo.load_state_dict(torch.load('saved_models/ppo_v3_'+str(args.lambd)+'.pt', map_location=torch.device('cpu')))


    for t in tqdm(range((max_steps-1)*n_demos)):
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

    pickle.dump(dataset, open('demonstrations/ppo_demos_v3_'+str(args.lambd)+'.pk', 'wb'))

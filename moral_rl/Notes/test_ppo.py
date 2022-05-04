from moral.ppo import *
from envs.gym_wrapper import *

import torch
from tqdm import tqdm
import wandb
import argparse
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np



# observation_space = spaces.Box(
#             low=0, high=1,
#             shape=(2, 2, 2),
#             dtype=np.int32
#         )
# n_actions = 4 # actions possibles dans chaque état
# obs_shape = observation_space.shape
# state_shape = obs_shape[:-1]
# in_channels = obs_shape[-1]

# print(obs_shape)
# print(state_shape)
# print(in_channels)
# ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
# optimizer = torch.optim.Adam(ppo.parameters(), lr=lr_ppo)
# ts = torch.tensor([[1.,1.],[-1.,-1.]]).to(device)
# overall_loss = torch.mean(ts)
# overall_loss.requres_grad = True
# optimizer.zero_grad()
# # overall_loss.backward()
# # optimizer.step()



# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# lr_ppo = 3e-4
# gamma = 0.999,
# epsilon = 0.1,
# ppo_epochs = 5
# entropy_reg = 0.05

# env = make_env('randomized_v1', 0)()
# states = env.reset()
# print("states = ",states)
# states_tensor = torch.tensor(states).float().to(device)

# n_actions = env.action_space.n
# print(n_actions)
# obs_shape = env.observation_space.shape
# print(obs_shape)
# state_shape = obs_shape[:-1]
# in_channels = obs_shape[-1]

# ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
# optimizer = torch.optim.Adam(ppo.parameters(), lr=lr_ppo)


# # # x = torch.randn(1, 1, requires_grad=True)
# # x = torch.tensor([[-0.0120]], requires_grad=True)
# # print(x)
# # lin = nn.Linear(1, 1) # your model or manual operations
# # out = lin(x)
# # print(out)
# # print(out.grad_fn)
# # optimizer.zero_grad()
# # out.backward()
# # print(out)
# # print(x)
# # optimizer.step()


# for t in tqdm(range(100)):
#         actions, log_probs = ppo.act(states_tensor)
#         next_states, rewards, done, info = env.step(actions)
#         scalarized_rewards = rewards

#         train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs, rewards)

#         if train_ready:
#             update_policy(ppo, dataset, optimizer, gamma, epsilon, ppo_epochs, entropy_reg=entropy_reg)

#         # Prepare state input for next time step
#         states = next_states.copy()
#         states_tensor = torch.tensor(states).float().to(device)





import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# print(net)

# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # # in your training loop:
# # optimizer.zero_grad()   # zero the gradient buffers

# # output = net(input)
# # loss = criterion(output, target)
# # loss.backward()
# optimizer.step()    # Does the update

# print("x = ", x)
# print("z = ", z)
# print("loss = ", loss)


# x = Variable(torch.FloatTensor([[1, 2, 3, 4]]), requires_grad=True)
# z = 2*x
# loss = z.sum(dim=1)
# print("loss = ", loss)

# # do backward for first element of z
# z.backward(torch.FloatTensor([[1, 0, 0, 0]]), retain_graph=True)
# print(x.grad.data)
# x.grad.data.zero_() #remove gradient in x.grad, or it will be accumulated

# # do backward for second element of z
# z.backward(torch.FloatTensor([[0, 1, 0, 0]]), retain_graph=True)
# print(x.grad.data)
# x.grad.data.zero_()

# # do backward for all elements of z, with weight equal to the derivative of
# # loss w.r.t z_1, z_2, z_3 and z_4
# z.backward(torch.FloatTensor([[1, 1, 1, 1]]), retain_graph=True)
# print(x.grad.data)
# x.grad.data.zero_()

# # or we can directly backprop using loss
# loss.backward() # equivalent to loss.backward(torch.FloatTensor([1.0]))
# print(x.grad.data) 

# a = torch.tensor([0.1], requires_grad = True)
# b = torch.tensor([1.0], requires_grad = True)
# c = torch.tensor([1.0], requires_grad = True)
a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
b = torch.tensor([3.0, 4.0, 5.0], requires_grad = True)
c = torch.tensor([6.0, 7.0, 8.0], requires_grad = True)

y=3*a + 2*b*b + torch.log(c)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# gradients = torch.FloatTensor([0.1])
y.backward(gradients)

# -> ce que ça fait : 
# -> 1/ dérive y par rapport à toutes les variables (requires_grad)
# ->      dy/da = 3
# ->      dy/db = 4b
# ->      dy/dc = 1/c
# -> 2/ applique le coefficient passé en argument de backward() aux dérivées partielles de chaque variable calculées en fonction des valeurs des variables :
# ->      [0.1, 1.0, 0.0001] * dy/da = [0.1, 1.0, 0.0001] * 3   = [0.1, 1.0, 0.0001] * 3                 = [3.0000e-01, 3.0000e+00, 3.0000e-04]
# ->      [0.1, 1.0, 0.0001] * dy/db = [0.1, 1.0, 0.0001] * 4b  = [0.1, 1.0, 0.0001] * 4*[3.0, 4.0, 5.0] = [1.2000e+00, 1.6000e+01, 2.0000e-03]
# ->      [0.1, 1.0, 0.0001] * dy/dc = [0.1, 1.0, 0.0001] * 1/c = [0.1, 1.0, 0.0001] * 1/[6.0, 7.0, 8.0] = [1.2000e+00, 1.6000e+01, 2.0000e-03]


print(a.grad) # tensor([3.3003])
print(b.grad) # tensor([0.])
print(c.grad) # tensor([inf])



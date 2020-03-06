import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import random
import numpy as np
from collections import deque
import math
import gym
from tensorboardX import SummaryWriter
from PointGoalNavigationEnv import *
import matplotlib.pyplot as plt
from prior_controller import *
from time import sleep
import ray
import scipy.stats as stats


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, mu, std


def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    w1 = 1
    w2 = 1
    mu = (np.power(policy_sigma, 2) * w1 * prior_mu + np.power(prior_sigma,2) * w2 * policy_mu)/(np.power(policy_sigma,2) * w1 + np.power(prior_sigma,2) * w2)
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma,2) * w1 + np.power(prior_sigma,2) * w2))
    return mu, sigma


def fuse_ensembles_stochastic(ensemble_actions):
    
    mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(num_agents)]), axis=0))/num_agents
    var = (np.sum(np.array([(ensemble_actions[i][1]**2 + ensemble_actions[i][0]**2)-mu**2 for i in range(num_agents)]), axis=0))/num_agents
    sigma = np.sqrt(var)
    return [mu, sigma]

def fuse_ensembles_deterministic(ensemble_actions):
    #print('Fusing emsembles')
    global num_agents
    actions = torch.tensor([ensemble_actions[i][0] for i in range (num_agents)])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

def compute_bhattacharyya_distance(mu_prior, sigma_prior, mu_policy, sigma_policy):
    
    dist = 0.25*math.log(0.25*((sigma_prior**2/(sigma_policy**2) + sigma_policy**2/(sigma_prior**2) + 2))) + 0.25*((mu_prior-mu_policy)**2/(sigma_prior**2 + sigma_policy**2))
    return dist

def rescale_action(action):
    return action * (action_range[1] - action_range[0]) / 2.0 +\
        (action_range[1] + action_range[0]) / 2.0

@ray.remote
def get_action(state, policy):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    a, _, mu, std = policy(state, False, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def get_action_simple(state, policy):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    a, _, mu, std = policy(state, False, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def test_env():
    print('Evaluating...')

    if VIS_GRAPH:
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
        plt.axis([-10,10,0,2])

    total_rewards = 0
    total_steps = 0
    num_episodes = 15
 
    for i in range(num_episodes):

        state = env.reset(i)
        if VIS: env.render()
        done = False
        
        while True:

            if METHOD=="hybrid":

                dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = env._get_position_data()
                mu_prior = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)

                laser_scans = [laser_scan + np.random.normal(0,0.5,180) for _ in range(100)]
                prior_actions = [prior.computeResultant(dist_to_goal, angle_to_goal, ls) for ls in laser_scans]
                prior_std_est =  np.std(prior_actions, 0)
                
                ensemble_actions = ray.get([get_action.remote(state, p) for p in policy_net_ensemble])
                mu_ensemble, std_ensemble = fuse_ensembles_deterministic(ensemble_actions)
                mu_hybrid, std_hybrid = fuse_controllers(mu_prior, std_prior, mu_ensemble, std_ensemble)

                vmu_policy = mu_ensemble[0]
                vsigma_policy = std_ensemble[0]

                vmu_prior = mu_prior[0]
                vsigma_prior = std_prior

                wmu_policy = mu_ensemble[1]
                wsigma_policy = std_ensemble[1]

                wmu_prior = mu_prior[1]
                wsigma_prior = std_prior  

                vmu_combined = mu_hybrid[0]
                vsigma_combined = std_hybrid[0]
                wmu_combined = mu_hybrid[1]
                wsigma_combined = std_hybrid[1]

                dist_combined  = Normal(torch.tensor([vmu_combined, wmu_combined]), torch.tensor([vsigma_combined, wsigma_combined]))

                if VIS_GRAPH:
                    x = np.linspace(-3, 3, 100)
                    plt.subplot(211)
                    plt.plot(x, stats.norm.pdf(x, vmu_policy, vsigma_policy))
                    plt.plot(x, stats.norm.pdf(x, vmu_prior, vsigma_prior))
                    plt.plot(x, stats.norm.pdf(x, vmu_combined, vsigma_combined))
                    plt.legend(['Policy', 'Prior', 'Combined'], loc="upper right")
                    plt.xlabel('Linear Velocity')
                    plt.subplot(212)
                    plt.plot(x, stats.norm.pdf(x, wmu_policy, wsigma_policy))
                    plt.plot(x, stats.norm.pdf(x, wmu_prior, wsigma_prior))
                    plt.plot(x, stats.norm.pdf(x, wmu_combined, wsigma_combined))
                    plt.xlabel('Angular Velocity')
                    fig.canvas.draw()
                    fig.clf()

                dist_combined  = Normal(torch.tensor(mu_hybrid), torch.tensor(std_hybrid))
                act = dist_combined.rsample()
                act = torch.tanh(act).numpy()
                
            elif METHOD == "policy":
                action_dist = get_action_simple(state, policy_net)
                dist_policy = Normal(torch.tensor(action_dist[0]), torch.tensor(action_dist[1]))
                act = dist_policy.sample()
                act = torch.tanh(act).numpy()
                

            elif METHOD == "prior":
                dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = env._get_position_data()

                laser_scans = [laser_scan + np.random.normal(-2,100,180) for _ in range(100)]
                prior_actions = [prior.computeResultant(dist_to_goal, angle_to_goal, ls) for ls in laser_scans]
                prior_std_est =  np.std(prior_actions, 0)
                print('Std: ', prior_std_est)

                mu_prior = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
                act = mu_prior

            elif METHOD == "random":
                act = np.random.rand(2) * 2 - 1


            next_state, reward, done, _ = env.step(act)
        
            state = next_state
            total_rewards += reward
            total_steps += 1

            if VIS: env.render()

            if done:
                
                break

#==========================================================================================================================

ray.init()
device = torch.device("cpu")
METHOD = "hybrid"
ENV = "PointGoalNavigation"
REWARD_TYPE = "sparse"
ALGORITHM = "SAC"
time_tag = str(time.time())
VIS = True
VIS_GRAPH = False
seed = 2

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

#==========================================================================================================================
# ENV PARAMS

HYPERS = dict(# training params
              viz                = "--viz" in sys.argv,
              log                = "--log" in sys.argv,
              use_cuda           = "--cuda" in sys.argv,
              load               = str  (sys.argv[sys.argv.index("--load"   )+1]) if "--load"    in sys.argv else None,
              runtime            = float(sys.argv[sys.argv.index("--runtime")+1]) if "--runtime" in sys.argv else None,
              ckpt_interval      = 10000,
              log_interval       = 1000,
              eval_interval      = 10000,

              # env params
              num_beams          = 180,
              laser_range        = 1.5,
              laser_noise        = 0.01,
              angle_min          = -np.pi/2,
              angle_max          = np.pi/2,
              timeout            = 500,
              velocity_max       = 1,
              omega_max          = 1,
              env_type           = 5,
              reward_type        = "sparse"
                
)

for k,v in HYPERS.items(): exec("{} = {!r}".format(k,v))

#==========================================================================================================================

env = PointGoalNavigation(**HYPERS)
env.seed(seed)
action_range = [env.action_space.low, env.action_space.high]

std_prior = 0.3

num_agents = 10
ensemble_file_name = "trained_ensemble_for_robot_new_5/1582454594.5134897_PointGoalNavigation_sparse_SAC_spinup_long_horizon_hybridFOR_ROBOT10_"

policy_net_ensemble = [torch.load(ensemble_file_name + str(i) + "_.pth").cpu() for i in range(num_agents)]
policy_net = policy_net_ensemble[0]
prior = PotentialFieldsController()

test_env()
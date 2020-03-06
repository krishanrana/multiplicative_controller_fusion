
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sac_core as core
from PointGoalNavigationEnv import *
from torch.utils.tensorboard import SummaryWriter
import ray
from torch.distributions import Normal
from prior_controller import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse
import cv2

ray.init()

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(agent.device) for k,v in batch.items()}



class SACAgent:

    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_fn = env_fn
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.timestep = 0
        self.a_lr = 3e-4

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)


        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        
        # Entropy temperature
        self.alpha = alpha

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, _ = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi, _, _ = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, logp_pi.detach()


    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def get_action(self, o, deterministic=False):
        act, mu, std = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
        return act, mu, std


def evaluate_agent(steps):
    
    # We are evaluating the policy as a stand-alone system in this case
    global agents
    global test_steps
    global total_steps
    total_reward = 0
    total_length = 0
    env.env_type = 2
    for _ in range(5):
        o = env.reset()
        d = False
        while not d:
            # Take deterministic actions at test time
            agnts = [i.ac for i in agents]
            ensemble_actions = ray.get([get_distr.remote(o, p) for p in agnts])
            mu, sigma = fuse_ensembles_deterministic(ensemble_actions)
            #print(mu, sigma)
            dist = Normal(torch.tensor(mu.detach()), torch.tensor(sigma.detach()))
            act = dist.sample()
            act = torch.tanh(act).numpy()
            o, r, d, _ = env.step(act)
            writer.add_scalar('{}/epistemic_uncertainty_vel'.format(ENV), sigma[0], test_steps)
            writer.add_scalar('{}/epistemic_uncertainty_omega'.format(ENV), sigma[1], test_steps)
            total_reward += r
            total_length += 1
            test_steps += 1
            if r == 1.0:
                break
    avg_rew = total_reward/5
    avg_len = total_length/5
    writer.add_scalar('{}/rewards_eval'.format(ENV), avg_rew, total_steps)
    writer.add_scalar('{}/length_eval'.format(ENV), avg_len, total_steps)
    

@ray.remote(num_gpus=1)
def get_distr(state, agent):
    state = torch.FloatTensor(state).unsqueeze(0).cuda()
    act, mu, std = agent.act(state, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def fuse_ensembles_stochastic(ensemble_actions):
    global num_agents
    mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(num_agents)]), axis=0))/num_agents
    var = (np.sum(np.array([(ensemble_actions[i][1]**2 + ensemble_actions[i][0]**2)-mu**2 for i in range(num_agents)]), axis=0))/num_agents
    sigma = np.sqrt(var)
    return mu, sigma

def fuse_ensembles_deterministic(ensemble_actions):
    #print('Fusing emsembles')
    global num_agents
    actions = torch.tensor([ensemble_actions[i][0] for i in range (num_agents)])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

def save_ensemble():
    global agents
    for idx, agnt in enumerate(agents):
        torch.save(agnt.ac.pi, save_dir + model_name + "_" + str(idx) + "_.pth")

def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma, alpha):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    if alpha > 1.0:
        alpha = 1.0
    alpha_comp = 1.0-alpha
    mu = (np.power(policy_sigma, 2) * alpha * prior_mu + np.power(prior_sigma,2) * w2 * policy_mu)/(np.power(policy_sigma,2) * alpha + np.power(prior_sigma,2) * alpha_comp)
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma,2) * alpha + np.power(prior_sigma,2) * alpha_comp))
    return mu, sigma

def alpha_exponential_decay_function(start_val=1, end_val=0.001, iterns=500000):
    rate = (end_val/start_val)**(1/iterns)
    return rate

def alpha_reverse_sigmoid_function(k, x0, x):
    val = 1 / (1 + math.exp(k(x - x0))) 
    return val

def gather_experience(agent, env, max_steps, ep):
    # Prepare for interaction with environment
    global total_steps
    global replay_buffer
    global agents
    global alpha

    env.env_type = np.random.choice([1,2,3,4,5])
    o, ep_ret, ep_len = env.reset(), 0, 0


    # Main loop: collect experience in env and update/log each epoch
    for t in range(max_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if total_steps > agent.start_steps:

            action, mu_policy, std_policy = agent.get_action(o)
            writer.add_scalar('{}/policy_mu_vel'.format(ENV), mu_policy[0], total_steps)
            writer.add_scalar('{}/policy_sigma_vel'.format(ENV), std_policy[0], total_steps)
            writer.add_scalar('{}/policy_mu_omega'.format(ENV), mu_policy[1], total_steps)
            writer.add_scalar('{}/policy_sigma_omega'.format(ENV), std_policy[1], total_steps)
            

            if METHOD == "baseline":
                if ep%2 == 0:
                    dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = env._get_position_data()
                    a = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
                else:
                    a = action
                    

            if METHOD == "hybrid":
                dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = env._get_position_data()
                mu_prior = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
                mu_hybrid, std_hybrid = fuse_controllers(mu_prior, std_prior, mu_policy.cpu().numpy(), std_policy.cpu().numpy(), alpha)
                dist_hybrid = Normal(torch.tensor(mu_hybrid), torch.tensor(std_hybrid))
                a = dist_hybrid.rsample()
                a = torch.tanh(a)

                writer.add_scalar('{}/prior_mu_vel'.format(ENV), mu_prior[0], total_steps)
                writer.add_scalar('{}/prior_sigma_vel'.format(ENV), std_prior, total_steps)
                writer.add_scalar('{}/prior_mu_omega'.format(ENV), mu_prior[1], total_steps)
                writer.add_scalar('{}/prior_sigma_omega'.format(ENV), std_prior, total_steps)

                writer.add_scalar('{}/hybrid_mu_vel'.format(ENV), mu_hybrid[0], total_steps)
                writer.add_scalar('{}/hybrid_sigma_vel'.format(ENV), std_hybrid[0], total_steps)
                writer.add_scalar('{}/hybrid_mu_omega'.format(ENV), mu_hybrid[1], total_steps)
                writer.add_scalar('{}/hybrid_sigma_omega'.format(ENV), std_hybrid[1], total_steps)

                alpha = alpha * alpha_decay


                if VIS_GRAPH:
                    x = np.linspace(-3, 3, 100)
                    fig = plt.figure(1)
                    plt.subplot(211)
                    plt.plot(x, stats.norm.pdf(x, mu_policy[0].cpu(), std_policy[0].cpu()))
                    plt.plot(x, stats.norm.pdf(x, mu_prior[0], std_prior))
                    plt.plot(x, stats.norm.pdf(x, mu_hybrid[0], std_hybrid[0]))
                    plt.legend(['Policy', 'Prior', 'Combined'], loc="upper right")
                    plt.xlabel('Linear Velocity')
                    plt.subplot(212)
                    plt.plot(x, stats.norm.pdf(x, mu_policy[1].cpu(), std_policy[1].cpu()))
                    plt.plot(x, stats.norm.pdf(x, mu_prior[1], std_prior))
                    plt.plot(x, stats.norm.pdf(x, mu_hybrid[1], std_hybrid[1]))
                    plt.xlabel('Angular Velocity')
                    #plt.show()
                    writer.add_figure('{}/dist'.format(ENV), fig, global_step=total_steps, close=True, walltime=None)

            if METHOD == "policy":
                a = action
                
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        total_steps += 1

        writer.add_scalar('{}/alpha'.format(ENV), alpha, total_steps)

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        if replay_buffer.size > agent.batch_size:
            for ag in agents:
                batch = replay_buffer.sample_batch(agent.batch_size)
                ag.update(data=batch)

        if d or t == agent.max_ep_len-1:
            writer.add_scalar('{}/ep_rewards'.format(ENV), ep_ret, total_steps)
            writer.add_scalar('{}/ep_length'.format(ENV), ep_len, total_steps)
            break

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

    return


#=========================================================================================================================================================#

parser = argparse.ArgumentParser(description='Parameters for training')
parser.add_argument('--method', type=str, default="hybrid", help="options include: policy, hybrid")
parser.add_argument('--env_type', type=int, default=5)
parser.add_argument('--seed', type=int, default=18)
parser.add_argument('--reward_type', type=str, default="sparse")
parser.add_argument('--colour', nargs='+', type=int, default=[255,222,0])
args = parser.parse_args()

ENV = "PointGoalNavigation"
REWARD_TYPE = args.reward_type
ALGORITHM = "SAC_spinup"
METHOD = args.method
SEED = args.seed
HORIZON = "long"
time_tag = str(time.time())
log_dir = "runs_video/" + time_tag + "_" + ENV + "_"  + REWARD_TYPE + "_" + ALGORITHM + "_" + HORIZON + "_horizon" + "_" + METHOD + "FOR_VIDEO" + str(SEED)
model_name = time_tag + "_" + ENV + "_" + REWARD_TYPE + "_" + ALGORITHM + "_" + HORIZON + "_horizon" + "_" + METHOD + "FOR_VIDEO" + str(SEED)
writer    = SummaryWriter(log_dir=log_dir)
save_dir = "pytorch_models/SAC_Data/" + "FOR_ROBOT_Ensemble_" + time_tag + '/'
os.mkdir(save_dir)
VIS_GRAPH = False
ep_len = 500

print('Method: ' + str(METHOD))
print('Reward Type: ' + str(REWARD_TYPE))
print('Seed: ' + str(SEED))

torch.set_num_threads(torch.get_num_threads())

# ENV PARAMS

HYPERS = dict(# training params
    # env params
    num_beams          = 180,
    laser_range        = 1.5,
    laser_noise        = 0.01,
    angle_min          = -np.pi/2,
    angle_max          = np.pi/2,
    timeout            = ep_len,
    velocity_max       = 1,
    omega_max          = 1,
    env_type           = 4,
    reward_type        = REWARD_TYPE,
    colour             = tuple(args.colour)
)

for k,v in HYPERS.items(): exec("{} = {!r}".format(k,v))

#=========================================================================================================================================================#

env = PointGoalNavigation(**HYPERS)
num_agents = 1
prior = PotentialFieldsController()
std_prior = 0.3

env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

agents = [SACAgent(lambda : env, 
                    actor_critic=core.MLPActorCritic,
                    ac_kwargs=dict(hidden_sizes=[256]*2), 
                    gamma=0.99, 
                    seed=i+SEED, 
                    steps_per_epoch=4000,
                    epochs=200,
                    replay_size=int(1e6),
                    polyak=0.995,
                    lr=1e-3,
                    alpha=0.1,
                    batch_size=128,
                    start_steps=10000,
                    update_after=1000,
                    update_every=50, 
                    num_test_episodes=10,
                    max_ep_len=ep_len,
                    save_freq=1) for i in range(num_agents)]

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))
num_episodes = 1200
total_steps = 0
test_steps = 0
alpha = 1
alpha_decay = alpha_exponential_decay_function(start_val=1, end_val=0.001, iterns=600000)



print("Training...")

for i in range(num_episodes):

    agent = random.choice(agents)
    gather_experience(agent, env, ep_len, i)
    
    if i%10 == 0:
        evaluate_agent(i)
        save_ensemble()

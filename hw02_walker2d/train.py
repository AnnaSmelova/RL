import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam

import random

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 2048

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

KL_DIFF = 0.02
MAX_GRAD_NORM = 0.5

ITERATIONS = 1000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.log_sigma = nn.Parameter(torch.zeros(action_dim)).to(DEVICE)
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.log_sigma)
        distribution = Normal(mu, sigma)
        probability = torch.exp(distribution.log_prob(action).sum(-1))
        return probability, distribution

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        # return None
        mu = self.model(state)
        sigma = torch.exp(self.log_sigma)
        distribution = Normal(mu, sigma)
        action = distribution.sample()
        action_transformed = torch.tanh(action)
        return action_transformed, action, distribution
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR, eps=1e-5)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float()  # Estimated by generalized advantage estimation
            
            # TODO: Update actor here
            # TODO: Update critic here
            new_probability, distribution = self.actor.compute_proba(s, a)

            kl = F.kl_div(new_probability, op)
            if kl > KL_DIFF:
                print(f"!!!!!! Big KL {kl} - BREAK")
                break

            ratios = torch.exp(torch.log(new_probability + 1e-8) - torch.log(op + 1e-8))
            s1 = ratios * adv
            s2 = torch.clip(ratios, 1 - CLIP, 1 + CLIP) * adv

            actor_loss = (-torch.min(s1, s2)).mean()
            actor_loss -= ENTROPY_COEF * distribution.entropy().mean()

            critic_v = self.critic.get_value(s).flatten()
            critic_loss = F.smooth_l1_loss(critic_v, v)

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)

            self.actor_optim.step()
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        #torch.save(self.actor, "agent.pkl")
        torch.save(self.actor.state_dict(), "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


def main():
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    lr = lambda f: f * ACTOR_LR
    #ppo.actor.load_state_dict(torch.load("agent.pkl"))  # дообучим еще
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    arewards = []
    alrs = []
    asteps = []
    rewards_max = -np.inf

    torch.save(ppo.actor.state_dict(), 'checkpoint_first.pth')

    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        if i > 0:
            frac = 1.0 - (i - 1.0) / 1000
            lrnow = lr(frac)
            ppo.actor_optim.param_groups[0]['lr'] = lrnow
            ppo.critic_optim.param_groups[0]['lr'] = lrnow

        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            rewards_mean = np.mean(rewards)
            asteps.append(steps_sampled)
            alrs.append(ppo.actor_optim.param_groups[0]['lr'])
            arewards.append(rewards_mean)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)} | "
                  f"Episodes: {episodes_sampled}, Steps: {steps_sampled}, LR: {ppo.actor_optim.param_groups[0]['lr']}")
            if rewards_mean > rewards_max:
                ppo.save()
                torch.save(ppo.actor.state_dict(), 'checkpoint_best.pth')
                rewards_max = rewards_mean

    return arewards, alrs, asteps


if __name__ == "__main__":
    main()

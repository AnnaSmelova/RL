from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from collections import deque, namedtuple
import random

GAMMA = 0.99
TAU = 1e-3  # for soft update of target parameters
INITIAL_STEPS = 2048
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)  # replay buffer size 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Пространство действий
    There are four discrete actions available: 
    1. do nothing
    2. fire left orientation engine
    3. fire main engine
    4. fire right orientation engine.

Пространство состояний
    The state is an 8-dimensional vector: 
    1-2. the coordinates of the lander in 'x' & 'y'
    3-4. its linear velocities in 'x' & 'y'
    5. its angle
    6. its angular velocity
    7-8. two booleans that represent whether each leg is in contact with the ground or not.
"""


class QNetwork(nn.Module):
    """Архитектура нейронной сети, приближающей Q-function"""

    def __init__(self, state_size, action_size, seed=0):
        """
        state_size (int): Размерность состояний
        action_size (int): Размерность действий
        seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return self.fc3(x)


class ReplayBuffer:
    """Буфер для хранения накопленного опыта"""

    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        """
        action_size (int): Размерность действий
        buffer_size (int): Максимальный размер буфера
        batch_size (int): Размер батча
        seed (int): Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # Используем очередь для буфера, элементы - именованные кортежи
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def consume_transition(self, transition):
        """Добавление нового опыта в память"""
        state, action, reward, next_state, done = transition
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample_batch(self):
        """Сэмплируем батч из буфера случайным образом"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim, seed=0):
        self.steps = 0  # Do not change
        self.model = QNetwork(state_dim, action_dim, seed).to(DEVICE)  # Torch model
        self.target_model = QNetwork(state_dim, action_dim, seed).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_buffer = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.consume_transition(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self.replay_buffer.sample_batch()

    def train_step(self, batch):
        # Use batch to update DQN's network.
        """
        batch (Tuple[torch.Variable]): Кортеж из кортежей (s, a, r, s', done)
        gamma (float): коэффициент GAMMA
        """
        if not self.model.training:
            self.model.train()
        # Распаковываем наш батч
        states, actions, rewards, next_states, dones = batch

        q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        q_expected = self.model(states).gather(1, actions)

        ### MSE loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.model, self.target_model, TAU)

    def soft_update(self, model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        # self.target_model.load_state_dict(self.model.state_dict())
        self.soft_update(self.model, self.target_model, TAU)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        if self.model.training:
            self.model.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        return np.argmax(action_values.cpu().data.numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        # torch.save(self.model, "agent.pkl")
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    agent.model.eval()
    for j in range(episodes):
        done = False
        state = env.reset()[0]
        total_reward = 0.

        tries = 0
        while not done and tries < 1000:
            tries += 1
            state, reward, done, _, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    agent.model.train()
    return returns


def main():
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 1  # 0.1
    eps_end = 0.01
    eps_decay = 0.995
    state = env.reset()[0]
    arewards = []
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        dqn.consume_transition((state, action, reward, next_state, done))

        state = next_state if not done else env.reset()[0]

    rewards_max = -np.inf
    tries_max = 1000
    state = env.reset()[0]
    tries = 0

    torch.save(dqn.model.state_dict(), 'checkpoint_first.pth')

    for i in range(TRANSITIONS):
        tries += 1
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _, _ = env.step(action)
        dqn.update((state, action, reward, next_state, done))

        if done or tries > tries_max:
            state = env.reset()[0]
            tries = 0
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        else:
            state = next_state

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            rewards_mean = np.mean(rewards)
            arewards.append(rewards_mean)
            print(f"Step: {i + 1}, Reward mean: {rewards_mean}, Reward std: {np.std(rewards)}")
            if rewards_mean > rewards_max:
                dqn.save()
                torch.save(dqn.model.state_dict(), 'checkpoint_best.pth')
                rewards_max = rewards_mean
    return arewards


if __name__ == "__main__":
    main()

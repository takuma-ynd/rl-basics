#!/usr/bin/env python3
import random
import gym
import gym.spaces
from gym.envs.registration import register
from collections import namedtuple
import numpy as np

import chainer
from chainer import Link, Chain
from chainer import Variable
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from tb_chainer import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

class Net(Chain):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(obs_size, hidden_size)
            self.l2 = L.Linear(hidden_size, n_actions)

    def forward(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()

    while True:
        obs_v = Variable(obs[np.newaxis, :].astype(np.float32))
        act_probs_v = F.softmax(net(obs_v))
        act_probs = act_probs_v.data[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))

            # reset variables
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    random.seed(12345)
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'is_slippery': False},
        max_episode_steps=100,
        )
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(gym.make('FrozenLakeNotSlippery-v0'))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net.cleargrads()  # zeros gradients
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(net)

    writer = SummaryWriter(comment='-frozenlake-nonslippery')

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        reward_mean = float(np.mean([s.reward for s in batch]))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue

        obs_v = Variable(np.asarray(obs, dtype=np.float32))
        acts_v = Variable(np.asarray(acts, dtype=np.int))
        full_batch = full_batch[-500:]

        action_scores_v = net(obs_v)
        loss_v = F.softmax_cross_entropy(action_scores_v, acts_v)
        loss_v.backward()

        optimizer.update()
        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, length of batch=%d" % (
            iter_no, loss_v.data, reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.data, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)

        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()

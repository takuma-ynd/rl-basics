#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np

import chainer
from chainer import Function, Variable
from chainer import Link, Chain, ChainList
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from tb_chainer import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

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
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = Variable(np.asarray(train_obs, dtype=np.float32))
    train_act_v = Variable(np.asarray(train_act, dtype=np.int32))
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print("obs_space:%d\tn_actions:%d" % (obs_size, n_actions))

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net.cleargrads()  # zero_grad
    optimizer = optimizers.Adam(alpha=0.01)
    optimizer.setup(net)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # net.cleargrads()  # zero_grad
        action_scores_v = net(obs_v)
        loss_v = F.softmax_cross_entropy(action_scores_v, acts_v)
        loss_v.backward()

        optimizer.update()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.data, reward_m, reward_b))
        writer.add_scalar("loss", loss_v.data, iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 199:
            print("Solved!")
            break
    writer.close()

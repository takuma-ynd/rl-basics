#!/usr/bin/env python3

from lib import wrappers
from lib import dqn_model_chainer as dqn_model

import argparse
import time
import numpy as np
import collections
import copy

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import backend
from chainer import backends
from chainer import Variable
from chainer import serializers

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        assert len(states) == len(actions) == len(rewards) == batch_size

        return states, actions, rewards, dones, next_states


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net , epsilon=0.0, device="cpu"):
        done_reward = None

        # perform epsilon-greedy policy
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

        else:
            xp = net.xp
            # predict the best action from state with q-network
            states_v = xp.array([self.state], copy=False)
            states_v = Variable(states_v)  # .to_device(device)
            q_vals_v = net(states_v)
            act_v = F.argmax(q_vals_v)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # append to replay buffer
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net , tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    xp = net.xp  # logic to switch numpy vs cupy

    states_v = Variable(xp.asarray(states))
    next_states_v = Variable(xp.asarray(next_states))
    actions_v = Variable(xp.asarray(actions))
    rewards_v = Variable(xp.asarray(rewards, dtype=xp.float32))
    done_mask = xp.asarray(dones, dtype=bool)

    state_action_values = F.select_item(net(states_v), actions_v)
    next_state_values = F.max(tgt_net(next_states_v), axis=1)
    # IMPORTANT: set value of zeros for dones
    next_state_values = F.where(done_mask,
                                xp.full(next_state_values.shape, 0.0, dtype=xp.float32),
                                next_state_values)
    next_state_values.unchain()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return F.mean_squared_error(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable cuda')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME, help='Name of the environment, default=' + DEFAULT_ENV_NAME)
    parser.add_argument('--reward', default=MEAN_REWARD_BOUND, type=float, help='Mean reward boundary for stop of training, default=%.2f' % MEAN_REWARD_BOUND)
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    args = parser.parse_args()

    # chainer's way to get and use device
    device = chainer.get_device(args.device)
    device.use()

    env = wrappers.make_env(args.env)

    # define networks
    net = dqn_model.DQN(env.action_space.n).to_device(device)
    tgt_net = dqn_model.DQN(env.action_space.n).to_device(device)
    net.cleargrads()
    writer = SummaryWriter(comment='-' + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(net)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:  # When the hell this can be None???
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])

            print('%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s' % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_mean_of_100', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)

            if best_mean_reward is None or mean_reward > best_mean_reward:
                # save the model
                if best_mean_reward is not None:
                    serializers.save_npz('dqn_pong_chainer.npz', net)
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net = copy.deepcopy(net)  # chainer's way to copy network weights

        # optimizer.cleargrads()
        net.cleargrads()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.update()
    writer.close()

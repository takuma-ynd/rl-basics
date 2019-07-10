#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import chainer
from chainer import Variable
from chainer import serializers

from lib import wrappers
from lib import dqn_model_chainer

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize', help="disable visualization of game play")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model_chainer.DQN(env.action_space.n)
    serializers.load_npz('dqn_pong_chainer.npz', net)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        xp = net.xp
        if args.visualize:
            env.render()
        state_v = Variable(xp.array([state], copy=False))
        q_vals = net(state_v).array
        action = xp.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)
    if args.record:
        env.env.close()

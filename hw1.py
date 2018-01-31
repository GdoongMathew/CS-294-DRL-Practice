#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


their_data_path = None

def view_expert():

    global their_data_path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    their_data_path = args.envname + "-expert.txt"

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        steps_numbers = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns),
                       'steps': np.array(steps_numbers)}
        # print('expert_data', expert_data)
    pickle.dump(expert_data, open(their_data_path, 'wb'))

def train_my_model():

    global their_data_path
    expert_data = pickle.load(open(their_data_path, 'rb'))
    print(expert_data)


if __name__ == '__main__':
    view_expert()
    train_my_model()

#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

"""

import pickle
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import tf_util
import gym
import load_policy


their_data_path = None


def view_expert(policy, data_path):

    policy_fn = policy
    their_data_path = data_path

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

            print(observations)
            print(actions)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns)}
        # print('expert_data', expert_data)
    pickle.dump(expert_data, open(their_data_path, 'wb'))


def simple_model(data):
    # first simple trainning model
    key_list = list(data.keys())
    observations = []
    actions = []
    for key in key_list:
        if key == 'observations':
            observations = data[key]
        elif key == 'actions':
            actions = data[key]

    # tuple to numpy array
    observations = np.array(observations)
    actions = np.array(actions)
    # reshape action array's shape from Nx1x3 to Nx3
    _actions = np.reshape(actions, (actions.shape[0], actions.shape[2]))

    # params
    learning_rate = 0.025
    train_iter = 10
    batch_size = 100  # how many datas contained in one batch
    num_batches = int(observations.shape[0] / batch_size)
    dropout_rate = 0.9

    # create tensorflow placeholder
    input_data = tf.placeholder(tf.float32, shape=[None, observations.shape[1]])
    output_data = tf.placeholder(tf.float32, shape=[None, _actions.shape[1]])

    # fully connected NN
    hidden = fully_connected(input_data, 3, activation_fn=tf.nn.relu)
    prediction = fully_connected(hidden, _actions.shape[1], activation_fn=None)

    # create input pipe line
    input_que = tf.train.slice_input_producer([observations, _actions], shuffle=False)
    batch_input = tf.train.batch(input_que, batch_size=batch_size)

    cost = tf.reduce_mean(tf.square(tf.subtract(output_data, prediction)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    coord = tf.train.Coordinator()

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print('Initialized global variable...')
        threads = tf.train.start_queue_runners(coord=coord)

        for train_step in range(train_iter):

            avg_cost = 0

            for i in range(num_batches):

                batch_observ, batch_act = sess.run(batch_input)
                print('batch number', i, "/", num_batches)

                _, c = sess.run([optimizer, cost], feed_dict={input_data: batch_observ, output_data: batch_act})

                avg_cost += c
            print('Average cost:', avg_cost)
            print('-----------------------------')

        coord.request_stop()
        coord.join(threads)
        global save_path
        save_path = saver.save(sess, '/hw1_model/hw_1_model.ckpt')
        print('Model saved in path: %s' % save_path)


def train_my_model():

    global their_data_path
    expert_data = pickle.load(open(their_data_path, 'rb'))
    print('Expert model:')
    print(expert_data)

    x1 = np.array(expert_data['observations'])
    x2 = np.array(expert_data['actions'])

    print('observations:')
    print(x1.shape)
    print('actions')
    print(x2.shape)

    simple_model(expert_data)


if __name__ == '__main__':
    #view_expert()

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

    #view_expert(policy_fn, their_data_path)
    train_my_model()
    exit()

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
save_path = None
save_name = None


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
    learning_rate = 0.001
    train_iter = 20
    batch_size = 1000  # how many datas contained in one batch
    num_batches = int(observations.shape[0] / batch_size)
    dropout_rate = 0.9

    # create tensorflow placeholder
    input_data = tf.placeholder(tf.float32, shape=[None, observations.shape[1]])
    output_data = tf.placeholder(tf.float32, shape=[None, _actions.shape[1]])

    # fully connected NN

    with tf.variable_scope('NN'):
        hidden = fully_connected(input_data, 10, activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer)
        prediction = fully_connected(hidden, _actions.shape[1], activation_fn=None)

    # convert numpy array to tensor
    observations = tf.cast(observations, tf.float32)
    _actions = tf.cast(_actions, tf.float32)

    # create input pipe line
    input_que = tf.train.slice_input_producer([observations, _actions], shuffle=False)
    batch_input = tf.train.batch(input_que, batch_size=batch_size)

    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.square(tf.subtract(output_data, prediction)), name='cost')

    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    coord = tf.train.Coordinator()

    log_path = 'Tensorboard/'
    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print('Initialized global variable...')
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        for train_step in range(train_iter):

            avg_cost = 0

            for i in range(num_batches):

                batch_observ, batch_act = sess.run(batch_input)
                print('batch number', i, "/", num_batches)

                _, c = sess.run([optimizer, cost], feed_dict={input_data: batch_observ, output_data: batch_act})
                # writer.add_summary(c, train_step * num_batches + i)

                avg_cost += c
            print('Average cost:', avg_cost)
            print('-----------------------------')

        global save_path, save_name
        save_path = 'hw1_model/'
        save_name = 'hw1_model.ckpt'
        saver.save(sess, 'hw1_model/hw1_model.ckpt')
        print('Model saved in path: %s' % (save_path + save_name))

        coord.request_stop()
        coord.join(threads)
        return save_path, save_name


def train_my_model():

    global their_data_path
    expert_data = pickle.load(open(their_data_path, 'rb'))

    simple_model(expert_data)


def restore_model():
    global save_path, save_name

    meta_path = None
    if not save_path:
        meta_path = 'hw1_model/hw1_model.ckpt.meta'
        meta_folder = 'hw1_model'
    else:
        meta_path = save_path + save_name + '.meta'
        meta_folder = save_path
    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, shape=[None, 11])

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(meta_folder))

        sess.run(tf.global_variables_initializer())
        all_var = tf.trainable_variables()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        for i in range(args.num_rollouts):

            obs = env.reset()
            done = False
            steps = 0
            totalr = 0

            while not done:
                action = tf_util.eval(all_var[0], feed_dict={input_data: obs[None, :]})

                print(action)

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
    #train_my_model()

    restore_model()
    exit()

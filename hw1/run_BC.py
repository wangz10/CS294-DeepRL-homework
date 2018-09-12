#!/usr/bin/env python  

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import tf_util
import gym
import load_policy
from behavior_cloning import BehavioralCloning

def describe_env(env):
    print('Observations:', env.observation_space.dtype, env.observation_space.shape)
    print('Actions:', env.action_space.dtype, env.action_space.shape)
    print('Action space high:', env.action_space.high)
    print('Action space low:', env.action_space.low)
    print('Time step limit:', env.spec.timestep_limit)
    return


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    # parser.add_argument('--n_hidden_units', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()

    print('Loading expert data for %s' % args.envname)
    expert_data = pickle.load(open(os.path.join('expert_data', args.envname + '.pkl'), 'rb'))

    observations = expert_data['observations']
    actions = expert_data['actions']
    returns_expert = expert_data['returns']

    print(observations.shape, actions.shape)

    env = gym.make(args.envname)
    describe_env(env)
    max_steps = env.spec.timestep_limit
    # Make n_hidden_units the same with observation_space
    n_hidden_units = env.observation_space.shape[0]

    bc = BehavioralCloning(env.observation_space.shape[0],
        env.action_space.shape[0],
        n_hidden_units=n_hidden_units
        )

    n_epochs = args.n_epochs
    batch_size = 32
    n_samples = observations.shape[0]
    n_batches = n_samples // batch_size
    sample_idx = np.arange(n_samples)

    print('Training BC model')
    for i in range(n_epochs):
        for ii in range(n_batches):
            start_idx = ii*batch_size
            end_idx = (ii+1)*batch_size
            observations_batch = observations[start_idx:end_idx]
            actions_batch = actions[start_idx:end_idx, 0]
            loss = bc.fit(observations_batch, actions_batch)

        print('Epoch %d loss = %.6f' % (i, loss))
        # shuffle samples
        np.random.shuffle(sample_idx)


    print('Evaluating policy from the trained BC')
    returns_bc = []
    for i in range(args.num_rollouts):
        sys.stdout.flush()
        print('iter {}/{}'.format(i, args.num_rollouts), end=" ")
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = bc.predict(obs.reshape(1, -1))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1

            if steps >= max_steps:
                break
        returns_bc.append(totalr)
        

    results_df = pd.DataFrame({
        'expert_returns': returns_expert,
        'bc_returns': returns_bc
        })

    print(results_df.describe())
    results_df.to_csv(
        os.path.join('results', '%s-%d-%d.csv' % \
            (args.envname, n_hidden_units, n_epochs)))


if __name__ == '__main__':
    main()
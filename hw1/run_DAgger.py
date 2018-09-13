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
from imitation_learning import DAgger
from utils import *

# def train(model, x, y, n_epochs=10, batch_size=32, shuffle=True):
#     '''Train a supervised ML model using mini-batch.
#     '''
#     n_samples = x.shape[0]
#     n_batches = n_samples // batch_size
#     sample_idx = np.arange(n_samples)

#     for i in range(n_epochs):
#         for ii in range(n_batches):
#             start_idx = ii*batch_size
#             end_idx = (ii+1)*batch_size
#             x_batch = x[start_idx:end_idx]
#             y_batch = y[start_idx:end_idx]
#             loss = model.fit(x_batch, y_batch)

#         print('Epoch %d loss = %.6f' % (i, loss))
#         if shuffle:
#             # shuffle samples
#             np.random.shuffle(sample_idx)
#             x = x[sample_idx]
#             y = y[sample_idx]
#     return 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_dagger_iters', type=int, default=10,
                        help='Number of DAgger iterations')

    args = parser.parse_args()
    envname = args.envname
    expert_policy_file = args.expert_policy_file
    num_rollouts = args.num_rollouts
    n_epochs = args.n_epochs
    n_dagger_iters = args.n_dagger_iters
    batch_size = 32

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    print('Loading expert data for %s' % envname)
    expert_data = pickle.load(open(os.path.join('expert_data', envname + '.pkl'), 'rb'))

    observations = expert_data['observations']
    actions = expert_data['actions']
    returns_expert = expert_data['returns']

    print(observations.shape, actions.shape)

    env = gym.make(envname)
    describe_env(env)
    max_steps = env.spec.timestep_limit
    # Make n_hidden_units the same with observation_space
    n_hidden_units = env.observation_space.shape[0]

    dagger = DAgger(env.observation_space.shape[0],
        env.action_space.shape[0],
        n_hidden_units=n_hidden_units
        )

    print('Training DAgger model using expert data')
    dagger.train(observations, actions[:, 0, :],
        n_epochs=n_epochs, batch_size=batch_size, shuffle=True)
    print('Evaluating DAgger policy...')
    returns = dagger.evaluate_policy(env, n_episodes=num_rollouts, max_steps=max_steps)
    print('average returns: %.4f' % np.mean(returns))

    results_df = pd.DataFrame({
        'expert_returns': returns_expert,
        'bc_returns': returns # The first round is essentially BC
        })

    for i in range(n_dagger_iters):
        print('DAgger iter %d' % i)
        print('Running learned policy and get annotated actions from the expert')
        D_pi, labeled_actions = dagger.run_policy_and_get_labeled_actions(env, policy_fn)
        print('Aggregating data...')
        observations = np.vstack((observations, D_pi))
        actions = np.vstack((actions, labeled_actions))
        print('Updating DAgger with aggregated data...')
        dagger.train(observations, actions[:, 0, :],
            n_epochs=n_epochs, batch_size=batch_size, shuffle=True)

        print('Evaluating DAgger policy...')
        returns = dagger.evaluate_policy(env, n_episodes=num_rollouts, max_steps=max_steps)
        results_df['dagger_returns-%d' % i] = returns
        print('average returns: %.4f' % np.mean(returns))


    print(results_df.describe())
    results_df.to_csv(
        os.path.join('results/DAgger', '%s-%d-%d-%d.csv' % \
            (envname, n_hidden_units, n_epochs, n_dagger_iters)))

if __name__ == '__main__':
    main()

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
from imitation_learning import BehavioralCloning
from utils import *


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
    bc.train(observations, actions[:, 0, :], n_epochs=n_epochs, 
        batch_size=batch_size, shuffle=True)

    print('Evaluating policy from the trained BC')
    returns_bc = bc.evaluate_policy(env, n_episodes=args.num_rollouts, 
        max_steps=max_steps)

    results_df = pd.DataFrame({
        'expert_returns': returns_expert,
        'bc_returns': returns_bc
        })

    print(results_df.describe())
    results_df.to_csv(
        os.path.join('results/BC', '%s-%d-%d.csv' % \
            (args.envname, n_hidden_units, n_epochs)))


if __name__ == '__main__':
    main()
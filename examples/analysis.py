"""Test the oscillator environment. Feel free to experiment with this code."""
import gym
import numpy as np
import oscillator  # pylint: disable=unused-import


Qs = [1, 2, 5, 10, 20]
fs = [1/10, 1/5, 1/2, 1, 2, 5, 10]
N = 50
results = np.zeros((N, len(Qs), len(fs)))

for i in range(N):
    for j, Q in enumerate(Qs):
        for k, f in enumerate(fs):
            print(i, Q, f)
            env = gym.make('Oscillator-v0', frequency=f, quality=Q)
            env.reset()
            done = False
            while not done:
                _, r, done, _ = env.step(env.action_space.sample())    # Random agent
                results[i, j, k] += r

np.save('tmp/oscillator_analysis.npy', results)

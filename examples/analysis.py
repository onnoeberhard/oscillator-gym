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
            env = gym.make('Oscillator-v0', mass=1/f, spring_constant=f, friction=1/Q)
            print(i, Q, f)
            env.reset()
            # env.render()
            done = False
            # t = 0
            while not done:
                _, r, done, _ = env.step(env.action_space.sample())    # Random agent
                results[i, j, k] += r
                # _, r, done, _ = env.step([-sin(t * f)])
                # _, r, done, _ = env.step([1])
                # env.render()
                # print("Reward:", r)
                # t += env.dt
np.save('tmp/oscillator_analysis.npy', results)

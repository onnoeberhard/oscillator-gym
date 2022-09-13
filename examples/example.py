"""Test the oscillator environment. Feel free to experiment with this code."""
from math import sin, pi

import gym
import oscillator  # pylint: disable=unused-import

# Create environment. Try also: target=None, or change stiffness, mass or friction.
env = gym.make('Oscillator-v0', initial_state=(1, 0))

# Play one episode
env.reset()
env.render()
done = False
t = 0
while not done:
    # _, r, done, _ = env.step(env.action_space.sample())    # Random agent
    # _, r, done, _ = env.step([0])                          # Unforced oscillator
    _, r, done, _ = env.step([-sin(t * 2*pi)])               # Perfect policy when starting at a postitive x value
    env.render()
    print("Reward:", r)
    t += env.dt

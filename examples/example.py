"""Test the oscillator environment. Feel free to experiment with this code."""
import gym
import oscillator  # pylint: disable=unused-import

# Create environment. Try also: target=None, or change spring_constant, mass or friction.
env = gym.make('Oscillator-v0', initial_state=(3, 0), target=4)

# Play one episode
env.reset()
env.render()
while True:
    _, r, done, _ = env.step(env.action_space.sample())    # Random agent
    # _, r, done, _ = env.step([0])                        # Unforced oscillator
    env.render()
    print("Reward:", r)
    if done:
        break

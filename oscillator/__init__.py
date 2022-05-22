"""Register oscillator environment with gym"""
import gym

from .oscillator import OscillatorEnv

gym.envs.register(
     id='Oscillator-v0',
     entry_point='oscillator:OscillatorEnv',
     max_episode_steps=1000,
)

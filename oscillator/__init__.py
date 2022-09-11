"""Simple driven damped harmonic oscillator gym environment"""
import gym

from .oscillator import OscillatorEnv

gym.envs.register(
     id='Oscillator-v0',
     entry_point='oscillator:OscillatorEnv',
)

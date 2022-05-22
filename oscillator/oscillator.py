"""Simple driven damped harmonic oscillator gym environment"""
from math import cos, exp, pi, sin, sqrt

import gym
import numpy as np
import pygame
from gym import spaces


class OscillatorEnv(gym.Env):
    """Simple driven damped harmonic oscillator gym environment"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, mass=1, spring_constant=1, friction=1, dt=1/30, target=5,
                 max_force=4*pi**2, initial_state=None):
        r"""Simple driven damped harmonic oscillator gym environment.

        The system is described by the following differential equation:
        .. math:: m\ddot x = F - b\dot x - kx

        Parameters
        ----------
        mass : float, optional
            Mass :math:`m` of the oscillator, by default 1.
        spring_constant : float, optional
            "Natural" spring constant :math:`k_{nat} = k / (4 \pi^2)` of the
            oscillator, by default 1.
            A natural spring constant of 1 with a mass of 1 leads to a
            period/frequency of 1.
        friction : float, optional
            Friction coefficient :math:`b` of the oscillator, by default 1.
        dt : float, optional
            Time step of the simulation, by default 1/30 (such that one time
            unit is 1s when rendering at 30 fps).
        target : float or None, optional
            Target position of the oscillator, by default 5.
            If not None, the episode finishes once the position (x = state[0])
            is >= target and the reward is sparse: 0 in every timestep and 1
            upon reaching the target.
            If None (or falsy), there is no target, and the reward is the
            power transferred into the system (energy of system after step -
            energy of system before step) / `dt`. In this case, the episode
            only finishes once the time limit is exceeded.
        max_force : float, optional
            Maximum force that can be applied to the oscillator, by default
            :math:`4 \pi^2`.
            The action space remains [-1, 1], but all actions are multiplied by
            `max_force`. The default value :math:`4 \pi^2` is chosen to be
            equal to the force exerted by the spring at position 1 with a
            natural spring constant of 1, i.e. action=1 and state=(1, 0) is
            an equilibrium when `spring_constant` is 1.
        initial_state : array_like or None, optional
            Initial state of the environment, by default None.
            If None, the initial state is sampled from a standard normal
            distribution.
        """
        # Oscillator configuration
        self.mass = mass
        self.spring_constant = spring_constant * 4 * pi**2
        self.friction = friction
        self.sigma = -self.friction / (2 * self.mass)
        self.omega = sqrt(self.spring_constant/self.mass - self.sigma**2)
        self.initial_state = initial_state
        if initial_state is not None:
            self.initial_state = np.asarray(initial_state, dtype=np.float32)
        else:
            self.initial_state = None

        # Check that configuration is underdamped
        valid_friction = 2 * sqrt(self.spring_constant * self.mass)
        assert self.friction <= valid_friction, (
            "With the current configuration, the oscillator system is overdamped. Please decrease the "
            f"friction coefficient to be below {valid_friction:f} to ensure an underdamped oscillator."
        )

        # Simulation parameters
        self.dt = dt

        # RL setup
        self.target = target
        self.max_force = max_force

        # Environment setup
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), (2,))
        self.action_space = spaces.Box(-1, 1, (1,))
        self.state = None
        self.energy = None
        self.max_energy = None

        # Rendering
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {'energy': self.energy}

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed, return_info=return_info, options=options)

        if self.initial_state is not None:
            self.state = self.initial_state
        else:
            self.state = self.np_random.standard_normal(2, dtype=np.float32)
        self.energy = (1/2 * self.mass * self.state[1]**2) + (1/2 * self.spring_constant * self.state[0]**2)
        self.max_energy = self.energy

        observation = self._get_obs()
        return (observation, self._get_info()) if return_info else observation

    def _update(self, action, state):
        action *= self.max_force
        c2 = state[0] - action/self.spring_constant
        c1 = (state[1] - self.sigma*c2) / self.omega
        state[0] = (exp(self.sigma * self.dt)
            * (c1*sin(self.omega * self.dt) + c2*cos(self.omega * self.dt)) + action/self.spring_constant)
        state[1] = (exp(self.sigma*self.dt) * ((self.sigma*c1 - self.omega*c2)*sin(self.omega * self.dt)
            + (self.sigma*c2 + self.omega*c1)*cos(self.omega * self.dt)))
        energy = (1/2 * self.mass * state[1]**2) + (1/2 * self.spring_constant * state[0]**2)
        return state, energy

    def step(self, action):
        action = min(max(action[0], -1), 1)

        # Update state
        prev_energy = self.energy
        self.state, self.energy = self._update(action, self.state)
        self.max_energy = max(self.max_energy, self.energy)

        # Compute reward
        if self.target:
            done = self.state[0] >= self.target
            reward = 1 if done else 0
        else:
            done = False
            reward = (self.energy - prev_energy) / self.dt

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    def render(self, mode="human"):
        W = 512
        H = W // 8

        if self.window is None and mode == "human":
            pygame.init()    # pylint: disable=no-member
            pygame.display.init()
            self.window = pygame.display.set_mode((W, H))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((W, H))
        canvas.fill((255, 255, 255))

        # Determine scale factor
        if self.target:
            target = self.target
        else:
            target = sqrt(2*self.max_energy / self.spring_constant)

        # Draw target
        if self.target:
            pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect((11*W / 12, 0), (W, H)))

        # Draw grid and origin
        for x in range(1, (int(target) if self.target else int(6/5 * target)) + 1):
            pygame.draw.line(canvas, 0, (x/target*5/12*W + W/2, 0), (x/target*5/12*W + W/2, H))
        for x in range(1, int(6/5 * target) + bool(target%5)):
            pygame.draw.line(canvas, 0, (-x/target*5/12*W + W/2, 0), (-x/target*5/12*W + W/2, H))
        pygame.draw.line(canvas, (255, 0, 0), (W/2, 0), (W/2, H))

        # Draw spring and mass
        pygame.draw.line(canvas, (0, 0, 255), (W/2, H/2), (self.state[0]/target*5/12*W + W/2, H/2))
        pygame.draw.circle(canvas, (0, 0, 255), (self.state[0]/target*5/12*W + W/2, H/2), 10)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:    # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()    # pylint: disable=no-member

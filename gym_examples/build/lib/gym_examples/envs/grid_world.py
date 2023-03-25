import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    The following dictionary maps abstract actions from `self.action_space` to 
    the direction we will walk in if that action is taken.
    I.e. 0 corresponds to "right", 1 to "up" etc.
    """
    _action_to_direction = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(self, render_mode=None, size=5, n_targets=1):
        self.size = size  # The size of the square grid
        assert n_targets <= size*size - 1
        self.n_targets = n_targets  # The number of targets in the grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        
        observation_space_dict = {"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int)}
        for i in range(1, n_targets+1):
            observation_space_dict[f"target_{i}"] = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        self.observation_space = spaces.Dict(observation_space_dict)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        observation = {"agent": self._agent_location}
        observation.update(self._target_locations)
        return observation

    def _get_info(self):
        return dict()
    
    @classmethod
    def transition_model(cls, obs, action, size):
        result_obs = obs.copy()
        agent_location = obs['agent']
        direction = cls._action_to_direction[action]
        agent_location = np.clip(
            agent_location + direction, 0, size - 1
        )
        result_obs['agent'] = agent_location
        done = cls.goal_state(result_obs)
        reward = -1
        return result_obs, reward, done, {}

    @staticmethod
    def goal_state(obs):
        agent_location = obs['agent']
        target_locations = [value for key, value in obs.items() if 'target' in key]
        return np.any([np.array_equal(agent_location, target_location) for target_location in target_locations], axis=0)

    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        choices = []
        for i in range(self.size):
            for j in range(self.size):
                choices.append([i, j])
        n_choices = self.size * self.size
        location_args = self.np_random.choice(n_choices, size=(1+self.n_targets,), replace=False)

        self._agent_location = np.array(choices[location_args[0]], dtype=int)
        self._target_locations = {f'target_{i}': np.array(choices[target_location_args], dtype=int) for i, target_location_args in enumerate(location_args[1:], 1)}



        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        done = np.any([np.array_equal(self._agent_location, target_location) for target_location in self._target_locations.values()])
        reward = -1
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld-v1")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for target_location in self._target_locations.values():
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
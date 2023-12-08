from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from snake_env import SnakeGame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os


class Agent(Env):
    mode: str

    def __init__(self, mode='train'):
        self.snake_game = SnakeGame()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.mode = mode

    def render(self):
        self.snake_game.view()

    def reset(self):
        del self.snake_game
        self.snake_game = SnakeGame()
        obs = self.snake_game.observe()
        return obs

    def step(self, action: int):
        self.snake_game.action(action, self.mode)
        obs = self.snake_game.observe()
        reward = self.snake_game.evaluate()
        done = self.snake_game.is_done()
        return obs, reward, done, {}


# Copied from documentation https://stable-baselines.readthedocs.io/en/master/guide/examples.html
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True

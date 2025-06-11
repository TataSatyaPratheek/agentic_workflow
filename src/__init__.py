# This file makes 'src' a Python package.

from gymnasium.envs.registration import register

# This line formally registers your custom environment with Gymnasium
# under the ID "Snake-v1".
register(
    id="Snake-v1",
    entry_point="src.game.snake_env:SnakeEnv",
)

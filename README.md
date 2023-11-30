# SnakeAi
Using reinforcement learning to play snake.

This project uses OpenAI Gym and Stable_Baseline3 to builds a custom environment and train a reinforcement learning model to play snake. Typically, OpenAI Gym has prebuilt environments for common Atari game, but this project constructs a custom environment to fully understand the steps used in reinforcement learning.

Right now the model runs for 100,000 steps for training

To train the model:
```
python main.py train <path_to_file>
```

To load a model and use:
```
python main.py load <path_to_file>
```

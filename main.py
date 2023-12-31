
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import argparse
from agent import Agent, SaveOnBestTrainingRewardCallback
import os
import gym


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
    title='Subcommands', dest='subcommand', help='Available subcommands')
train_parser = subparsers.add_parser('train', help='train a model')
train_parser.add_argument('fileName', type=str,
                          help='File to train the model with')

load_parser = subparsers.add_parser('load')
load_parser.add_argument('fileName', type=str,
                         help='File to train the model with')

if __name__ == '__main__':
    # Setting up params
    args = parser.parse_args()
    log_dir = "log/"
    os.makedirs(log_dir, exist_ok=True)

    if (args.subcommand == 'train'):
        # Training new model
        env = Monitor(env=Agent(args.subcommand), filename=log_dir)
        model = PPO('MlpPolicy', env, verbose=1,
                    tensorboard_log="./board/", learning_rate=0.00003)

        print('--------Starting Training--------')
        callback = SaveOnBestTrainingRewardCallback(
            5000, log_dir=log_dir)
        model.learn(total_timesteps=1000000, callback=callback)
        model.save(args.fileName)
        print('--------Finished Training--------')

    elif (args.subcommand == 'load'):
        env = Agent(args.subcommand)
        # Testing model
        model = PPO.load(args.fileName)
        episodes = 10
        for episode in range(1, episodes):
            obs, _info = env.reset()
            done = False
            score = 0
            env.render()

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                score += reward
                env.render()
            print('Episode:{} Score:{}'.format(episode, score))


from stable_baselines3 import PPO
import argparse
from agent import Agent


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
    args = parser.parse_args()
    env = Agent(args.subcommand)

    if (args.subcommand == 'train'):
        # Training new model
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save(args.fileName)
    elif (args.subcommand == 'load'):
        # Testing model
        model = PPO.load(args.fileName)
        episodes = 10
        for episode in range(1, episodes):
            obs = env.reset()
            done = False
            score = 0
            env.render()

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                score += reward
                env.render()
            print('Episode:{} Score:{}'.format(episode, score))

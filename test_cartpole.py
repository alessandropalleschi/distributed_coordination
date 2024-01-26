import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
from env import MultiRobotEnv

if __name__ == '__main__':
    env = MultiRobotEnv(num_obstacles=20,num_robots=1)
    N = 20
    batch_size = 250
    n_epochs = 20
    alpha = 0.0003
    agent = Agent(n_actions=2, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 300_000_000
    agent.load_models()
    figure_file = 'plots/learning.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, dones, info = env.step(action)
            done = dones.all()
            n_steps += 1
            score += reward

            for robot_id in range(env.num_robots):
                agent.store_transition(observation[robot_id], action[robot_id],
                                       prob[robot_id], val[robot_id], reward[robot_id], dones[robot_id],robot_id)
            observation = observation_
        agent.learn()    
        learn_iters += 1
        score = np.mean(score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        agent.scores = score
        agent.avg_scores = avg_score
        agent.best_score = best_score
        agent.log_statistics()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
from env import MultiRobotEnv

if __name__ == '__main__':
    env = MultiRobotEnv(num_obstacles=20,num_robots=1)
    N = 20
    batch_size = 250
    n_epochs = 20
    alpha = 0.0003
    agent = Agent(n_actions=2, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 300_000_000
    agent.load_models()
    figure_file = 'plots/learning.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, dones, info = env.step(action)
            done = dones.all()
            n_steps += 1
            score += reward

            for robot_id in range(env.num_robots):
                agent.store_transition(observation[robot_id], action[robot_id],
                                       prob[robot_id], val[robot_id], reward[robot_id], dones[robot_id],robot_id)
            observation = observation_
        agent.learn()    
        learn_iters += 1
        score = np.mean(score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        agent.scores = score
        agent.avg_scores = avg_score
        agent.best_score = best_score
        agent.log_statistics()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

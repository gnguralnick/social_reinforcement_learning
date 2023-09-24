from ray.rllib import MultiAgentEnv

from models.model import Model
from collections import defaultdict
import numpy as np
from gymnasium.spaces import flatten_space


def test(models: dict[str, Model], env: MultiAgentEnv, num_episodes, num_agents, eps, eps_decay_factor, render=False):
    total = 0
    num_actions = flatten_space(env.action_space).shape[0]
    for episode in range(1, num_episodes + 1):
        states, _ = env.reset()

        all_done = False
        score = defaultdict(int)

        while not all_done:
            if render:
                env.render()
            action_dict = {}
            for agent in range(num_agents):
                if np.random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    p = models[str(agent)].predict([states[str(agent)][0].reshape(1, num_agents, 2),
                                                    states[str(agent)][1].reshape(1, 25, 18)])
                    # print(states[str(agent)][0])
                    # print(states[str(agent)][0].reshape(1, num_agents_cleanup, 2))
                    action = np.argmax(p)
                # print(action)
                agent_id = str(agent)
                action_dict[agent_id] = action
            n_state, rewards, dones, _, info = env.step(action_dict)
            all_done = all(value for value in dones.values())
            states = n_state
            for key in rewards.keys():
                score[key] += rewards[key]
        print('Episode:{} Score:{}'.format(episode, score))
        score_sum = 0
        for s in score.values():
            score_sum += s
        total += score_sum / len(score)
        eps *= eps_decay_factor
        print("Average Agent Reward: {}".format(score_sum / len(score)))
    print("Total avg: {}".format(total / 10))
    env.close()
    return total


def test_centralized(model: Model, env: MultiAgentEnv, num_episodes, num_agents, eps, eps_decay_factor, render=False):
    total = 0
    num_actions = flatten_space(env.action_space).shape[0]
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        all_done = False
        score = defaultdict(int)

        while not all_done:
            if render:
                env.render()
            action_dict = {}
            a = model.predict([state["0"][0].reshape(1, num_agents, 2), state["0"][1].reshape(1, 25, 18)],
                              verbose=0)[0]
            # print(a)
            for agent in range(num_agents):
                if np.random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    action = np.argmax(a[agent])
                # print(action)
                agent_id = str(agent)
                action_dict[agent_id] = action
            n_state, rewards, dones, _, info = env.step(action_dict)
            all_done = all(value for value in dones.values())
            state = n_state
            for key in rewards.keys():
                score[key] += rewards[key]
        print('Episode:{} Score:{}'.format(episode, score))
        score_sum = 0
        for s in score.values():
            score_sum += s
        total += score_sum / len(score)
        print("Average Agent Reward: {}".format(score_sum / len(score)))
        eps *= eps_decay_factor
    print("Total avg: {}".format(total / 10))
    env.close()
    return total

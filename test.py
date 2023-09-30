from gymnasium import spaces
from ray.rllib import MultiAgentEnv

from models.model import Model
from collections import defaultdict
import numpy as np
from gymnasium.spaces import flatten_space


def test(models: dict[str, Model], env: MultiAgentEnv, num_episodes, eps, eps_decay_factor, render=False):
    stats = {str(i): [] for i in range(len(models))}
    stats['total'] = []
    stats['average'] = []
    num_actions = flatten_space(env.action_space).shape[0]
    num_agents = len(env.get_agent_ids())
    if isinstance(env.observation_space, spaces.Tuple):
        env_height = env.observation_space[1].shape[0]
        env_width = env.observation_space[1].shape[1]
    else:
        env_height = env.observation_space.shape[0]
        env_width = env.observation_space.shape[1]
    for episode in range(1, num_episodes + 1):
        states, _ = env.reset()

        all_done = False
        agent_rewards = defaultdict(int)
        episode_reward = 0

        for i in range(num_agents):
            stats[str(i)].append(dict())
            stats[str(i)][-1]['total_reward'] = 0
            stats[str(i)][-1]['rewards'] = []

        while not all_done:
            if render:
                env.render()
            action_dict = {}
            for agent in range(num_agents):
                if np.random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    p = models[str(agent)].predict([states[str(agent)][0].reshape(1, num_agents, 2),
                                                    states[str(agent)][1].reshape(1, env_height, env_width)])
                    action = np.argmax(p)
                agent_id = str(agent)
                action_dict[agent_id] = action
            n_state, rewards, dones, _, info = env.step(action_dict)
            all_done = all(value for value in dones.values())
            states = n_state
            for agent in range(num_agents):
                agent_id = str(agent)
                episode_reward += rewards[agent_id]
                agent_rewards[agent_id] += rewards[agent_id]
                stats[agent_id][-1]["rewards"].append(rewards[agent_id])
                stats[agent_id][-1]["total_reward"] += rewards[agent_id]
        
        stats['total'].append(episode_reward)
        stats['average'].append(episode_reward / num_agents)
        
        print("\rEpisode {}/{} (total reward: {})".format(episode + 1, num_episodes, episode_reward))
        print("Agent rewards: {}".format(agent_rewards))

        eps *= eps_decay_factor
    env.close()
    print(stats)
    return stats


def test_centralized(model: Model, env: MultiAgentEnv, num_episodes, eps, eps_decay_factor, render=False):
    stats = None
    num_actions = flatten_space(env.action_space).shape[0]
    num_agents = len(env.get_agent_ids())
    if isinstance(env.observation_space, spaces.Tuple):
        env_height = env.observation_space[1].shape[0]
        env_width = env.observation_space[1].shape[1]
    else:
        env_height = env.observation_space.shape[0]
        env_width = env.observation_space.shape[1]
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        if stats is None:
            stats = {str(i): [] for i in range(num_agents)}
            stats['total'] = []
            stats['average'] = []
        all_done = False
        
        agent_rewards = defaultdict(int)
        episode_reward = 0
        for i in range(num_agents):
            stats[str(i)].append(dict())
            stats[str(i)][-1]['total_reward'] = 0
            stats[str(i)][-1]['rewards'] = []

        while not all_done:
            if render:
                env.render()
            action_dict = {}
            a = model.predict([state["0"][0].reshape(1, num_agents, 2), state["0"][1].reshape(1, env_height, env_width)],
                              verbose=0)[0]
            for agent in range(num_agents):
                if np.random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    action = np.argmax(a[agent])
                agent_id = str(agent)
                action_dict[agent_id] = action
            n_state, rewards, dones, _, info = env.step(action_dict)
            all_done = all(value for value in dones.values())
            state = n_state
            for agent in range(num_agents):
                agent_id = str(agent)
                episode_reward += rewards[agent_id]
                agent_rewards[agent_id] += rewards[agent_id]
                stats[agent_id][-1]["rewards"].append(rewards[agent_id])
                stats[agent_id][-1]["total_reward"] += rewards[agent_id]
        
        stats['total'].append(episode_reward)
        stats['average'].append(episode_reward / num_agents)
        
        print("\rEpisode {}/{} (total reward: {})".format(episode + 1, num_episodes, episode_reward))
        print("Agent rewards: {}".format(agent_rewards))

        eps *= eps_decay_factor
    env.close()
    print(stats)
    return stats

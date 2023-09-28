from collections import defaultdict
import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

from models.model import Model
from gymnasium.spaces import flatten_space


def train(models: dict[str, Model], env: MultiAgentEnv, num_episodes, eps, eps_decay_factor, discount_factor, alpha,
          batch_size, render=False):
    stats = { str(i): [] for i in range(len(models))}
    stats['total'] = []
    stats['average'] = []
    num_actions = flatten_space(env.action_space).shape[0]
    if isinstance(env.observation_space, spaces.Tuple):
        env_height = env.observation_space[1].shape[0]
        env_width = env.observation_space[1].shape[1]
    else:
        env_height = env.observation_space.shape[0]
        env_width = env.observation_space.shape[1]
    for i_episode in range(num_episodes):
        states, _ = env.reset()
        terminate = False
        num_agents = len(states)
        for i in range(num_agents):
            stats[str(i)].append(dict())
            stats[str(i)][-1]['total_reward'] = 0
            stats[str(i)][-1]['rewards'] = []
        episode_reward = 0
        agent_reward = [0] * num_agents
        state_batches = defaultdict(list)
        pos_batches = defaultdict(list)
        target_batches = defaultdict(list)
        while not terminate:
            if render:
                env.render()
            actions = {}
            for i in range(num_agents):
                agent_id = str(i)
                state_batches[agent_id].append(states[agent_id][1])
                pos_batches[agent_id].append(states[agent_id][0])
                if np.random.random() < eps:
                    actions[agent_id] = np.random.randint(0, num_actions)
                else:
                    actions[agent_id] = np.argmax(models[agent_id].predict(
                        [states[agent_id][0].reshape(1, num_agents, 2), states[agent_id][1].reshape(1, env_height, env_width)]))
            new_states, rewards, done, _, _ = env.step(actions)
            for i in range(num_agents):
                agent_id = str(i)
                episode_reward += rewards[agent_id]
                agent_reward[i] += rewards[agent_id]
                stats[agent_id][-1]["rewards"].append(rewards[agent_id])
                stats[agent_id][-1]["total_reward"] += rewards[agent_id]
                if not models[agent_id].use_model:
                    continue
                target = rewards[agent_id] + discount_factor * np.max(models[agent_id].predict(
                    [new_states[agent_id][0].reshape(1, num_agents, 2),
                     new_states[agent_id][1].reshape(1, env_height, env_width)]))
                stats[agent_id][-1]["target"] = target
                target_vector = models[agent_id].predict(
                    [states[agent_id][0].reshape(1, num_agents, 2), states[agent_id][1].reshape(1, env_height, env_width)])[0]
                target_vector[actions[agent_id]] = (1 - alpha) * target_vector[actions[agent_id]] + alpha * target
                stats[agent_id][-1]["target_vector"] = target_vector
                target_batches[agent_id].append(target_vector)
            if len(state_batches["0"]) == batch_size:
                for i in range(num_agents):
                    agent_id = str(i)
                    if not models[agent_id].use_model:
                        continue
                    models[agent_id].fit(
                        [np.array(pos_batches[agent_id]).reshape(batch_size, num_agents, 2),
                         np.array(state_batches[agent_id]).reshape(batch_size, env_height, env_width)],
                        np.array(target_batches[agent_id]).reshape(batch_size, num_actions),
                        epochs=1, verbose=0)
                state_batches = defaultdict(list)
                pos_batches = defaultdict(list)
                target_batches = defaultdict(list)
            states = new_states
            terminate = done["__all__"]
        stats['total'].append(episode_reward)
        stats['average'].append(episode_reward / num_agents)
        eps *= eps_decay_factor
        print("\rEpisode {}/{} (total reward: {})".format(i_episode + 1, num_episodes, episode_reward))
        print("Agent rewards: {}".format(agent_reward))

    print(stats)
    return models, stats


def train_centralized(model: Model, env: MultiAgentEnv, num_episodes, eps, eps_decay_factor, discount_factor, alpha,
                      batch_size, render=False):
    stats = None
    num_actions = flatten_space(env.action_space).shape[0]
    if isinstance(env.observation_space, spaces.Tuple):
        env_height = env.observation_space[1].shape[0]
        env_width = env.observation_space[1].shape[1]
    else:
        env_height = env.observation_space.shape[0]
        env_width = env.observation_space.shape[1]
    for i_episode in range(num_episodes):
        states, _ = env.reset()
        if stats is None:
            stats = { str(i): [] for i in range(len(states)) }
            stats['total'] = []
            stats['average'] = []
        for i in range(len(states)):
            stats[str(i)].append(dict())
            stats[str(i)][-1]['total_reward'] = 0
            stats[str(i)][-1]['rewards'] = []
        terminate = False
        num_agents = len(states)
        episode_reward = 0
        agent_reward = [0] * num_agents
        state_batch = []
        pos_batch = []
        target_batch = []
        while not terminate:
            if render:
                env.render()
            actions = {}
            a = model.predict([states["0"][0].reshape(1, num_agents, 2), states["0"][1].reshape(1, env_height, env_width)],
                              verbose=0)[0]
            state_batch.append(states["0"][1])
            pos_batch.append(states["0"][0])
            for i in range(num_agents):
                if np.random.random() < eps:
                    actions[str(i)] = np.random.randint(0, num_actions)
                else:
                    actions[str(i)] = np.argmax(a[i])
            new_states, rewards, done, _, _ = env.step(actions)

            target_vector = a

            for i in range(num_agents):
                episode_reward += rewards[str(i)]
                agent_reward[i] += rewards[str(i)]
                stats[str(i)][-1]["rewards"].append(rewards[str(i)])
                stats[str(i)][-1]["total_reward"] += rewards[str(i)]
                if not model.use_model:
                    continue
                target = rewards[str(i)] + discount_factor * np.max(model.predict(
                    [new_states[str(i)][0].reshape(1, num_agents, 2), new_states[str(i)][1].reshape(1, env_height, env_width)],
                    verbose=0)[0][i])
                stats[str(i)][-1]["target"] = target
                target_vector[i][actions[str(i)]] = (1 - alpha) * target_vector[i][actions[str(i)]] + alpha * (target)
                stats[str(i)][-1]["target_vector"] = target_vector[i]
            target_batch.append(target_vector)
            if len(state_batch) == batch_size:
                model.fit(
                    [np.array(pos_batch).reshape(batch_size, num_agents, 2),
                     np.array(state_batch).reshape(batch_size, env_height, env_width)],
                    np.array(target_batch).reshape(batch_size, num_agents, 4),
                    epochs=1, verbose=0)
                state_batch = []
                pos_batch = []
                target_batch = []
            states = new_states
            terminate = done["__all__"]
        stats['total'].append(episode_reward)
        stats['average'].append(episode_reward / num_agents)
        print("\rEpisode {}/{} (total reward: {})".format(i_episode + 1, num_episodes, episode_reward))
        print("Agent rewards: {}".format(agent_reward))
        eps *= eps_decay_factor

    return model, stats

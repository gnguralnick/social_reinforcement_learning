from collections import defaultdict
import numpy as np
from ray.rllib.env import MultiAgentEnv

from models.model import Model

batch_size = 10


def train(models: dict[str, Model], env: MultiAgentEnv, num_episodes, eps, eps_decay_factor, discount_factor, alpha, render=False):
    rewards_graph = []
    for i_episode in range(num_episodes):
        states, _ = env.reset()
        terminate = False
        num_agents = len(states)
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
                    actions[agent_id] = np.random.randint(0, env.action_space.shape[0])
                else:
                    actions[agent_id] = np.argmax(models[agent_id].predict(
                        [states[agent_id][0].reshape(1, num_agents, 2), states[agent_id][1].reshape(1, 25, 18)]))
            new_states, rewards, done, _, _ = env.step(actions)
            for i in range(num_agents):
                agent_id = str(i)
                if not models[agent_id].use_model:
                    continue
                episode_reward += rewards[agent_id]
                agent_reward[i] += rewards[agent_id]
                target = rewards[agent_id] + discount_factor * np.max(models[agent_id].predict(
                    [new_states[agent_id][0].reshape(1, num_agents, 2),
                     new_states[agent_id][1].reshape(1, 25, 18)]))
                target_vector = models[agent_id].predict(
                    [states[agent_id][0].reshape(1, num_agents, 2), states[agent_id][1].reshape(1, 25, 18)])[0]
                target_vector[actions[agent_id]] = (1 - alpha) * target_vector[actions[agent_id]] + alpha * target
                target_batches[agent_id].append(target_vector)
            if len(state_batches["0"]) == batch_size:
                for i in range(num_agents):
                    agent_id = str(i)
                    if not models[agent_id].use_model:
                        continue
                    models[agent_id].fit(
                        [np.array(pos_batches[agent_id]).reshape(batch_size, num_agents, 2),
                         np.array(state_batches[agent_id]).reshape(batch_size, 25, 18)],
                        np.array(target_batches[agent_id]).reshape(batch_size, env.action_space.shape[0]),
                        epochs=1, verbose=0)
                state_batches = defaultdict(list)
                pos_batches = defaultdict(list)
                target_batches = defaultdict(list)
            states = new_states
            terminate = done["__all__"]
        rewards_graph.append(episode_reward)
        eps *= eps_decay_factor
        print("\rEpisode {}/{} (total reward: {})".format(i_episode + 1, num_episodes, episode_reward))
        print("Agent rewards: {}".format(agent_reward))
        print(rewards_graph)

    return models, rewards_graph

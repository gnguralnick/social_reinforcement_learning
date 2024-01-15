from environments.one_d_cleanup_env import OneDCleanupEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

num_episodes = 20
verbose_episode = num_episodes - 1
steps_per_episode = 1000
num_agents = 10
agent_ids = [str(i) for i in range(num_agents)]
thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05
dirt_multiplier = 10

area = 150
env = OneDCleanupEnv(agent_ids=agent_ids,
                     num_agents=num_agents,
                     area=area,
                     thresholdDepletion=thresholdDepletion,
                     thresholdRestoration=thresholdRestoration,
                     wasteSpawnProbability=wasteSpawnProbability,
                     appleRespawnProbability=appleRespawnProbability,
                     dirt_multiplier=dirt_multiplier, use_randomness=False)
test_stats = []
ending_ep_rewards = []
for episode in range(num_episodes):
    test_stats.append({
        "num_apples": [],
        "num_dirt": [],
        "pickers": [],
        "cleaners": [],
        "total_reward": 0,
    })

    print(f"========= Episode {episode} =========")

    states, info = env.reset()
    state = states["coordinator"]
    test_stats[-1]["num_apples"].append(info["apple"])
    test_stats[-1]["num_dirt"].append(info["dirt"])
    test_stats[-1]["pickers"].append(info["picker"])
    test_stats[-1]["cleaners"].append(info["cleaner"])

    for step in tqdm(range(steps_per_episode)):
        #env.render()
        num_apples, num_dirt, _, _ = state
        agent_frequency_in_dirt = num_dirt / (num_apples + num_dirt)
        num_cleaner = round(num_agents * agent_frequency_in_dirt)
        num_picker = num_agents - num_cleaner
        # num_cleaner = 6
        # num_picker = 4
        assignments = env.get_greedy_assignments(num_picker, num_cleaner)
        actions = env.get_greedy_actions(assignments)
        next_states, reward, dones, info = env.step(actions)
        next_state = next_states["coordinator"]

        test_stats[-1]["num_apples"].append(info["apple"])
        test_stats[-1]["num_dirt"].append(info["dirt"])
        test_stats[-1]["pickers"].append(info["picker"])
        test_stats[-1]["cleaners"].append(info["cleaner"])
        #reward = sum(rewards.values())

        state = next_state

        if episode > verbose_episode:
            print(f"========= Step {step} =========")
            print(f"info: {info}")

        if dones["__all__"]:
            break

    ending_reward = info["total_apple_consumed"]

    test_stats[-1]["total_reward"] = ending_reward
    ending_ep_rewards.append(ending_reward)

    print(f"Ending reward: {ending_reward}")
    #print(f"reward graph: {reward_graph}")
    print(f"========= End of Episode {episode} =========")
    print(ending_ep_rewards)

from environments import OneDCleanupEnv
import numpy as np
import torch
from agents import OneDUCoordinator
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda available: {torch.cuda.is_available()}")
np.set_printoptions(threshold=np.inf)

reward_multiplier = 1

# for printing options
pp = False
verbose = False
verbose_episode = 2000  # start printing at which epoch

# env param
num_agents = 10
thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05

# u-net param
dirt_multiplier = 1
division_ep = 1e-7
gamma = 0.999
epsilon = 0.1
epsilon_decay = 0.99995
epsilon_min = 0.05
lr = 0.0001
batch_size = 10
agent_ids = [str(i+1) for i in range(num_agents)]
state_dim = 4
num_roles = 2
env = OneDCleanupEnv(agent_ids=agent_ids,
                     num_agents=num_agents,
                     thresholdDepletion=thresholdDepletion,
                     thresholdRestoration=thresholdRestoration,
                     wasteSpawnProbability=wasteSpawnProbability,
                     appleRespawnProbability=appleRespawnProbability,
                     dirt_multiplier=dirt_multiplier,
                     area=150, use_randomness=False)

agentCoordinator = OneDUCoordinator(device=device,
                                    env=env,
                                    num_agents=num_agents,
                                    num_roles=num_roles,
                                    buffer_size=5000,
                                    batch_size=batch_size,
                                    lr=lr,
                                    gamma=gamma,
                                    epsilon=epsilon,
                                    epsilon_decay=epsilon_decay,
                                    epsilon_min=epsilon_min,
                                    u_layers=[
                                        (state_dim, 200),
                                        (200, 100),
                                        (100, 50),
                                        (50, 1)
                                    ])
ending_ep_rewards = []
num_episodes = 2000
steps_per_epsiode = 1000
max_reward = 0

model_saves = deque()

f_good = open("good_oned_u.txt", "w")  # record good results (set your threshold)

for episode in range(num_episodes):
    print(f"========= Episode {episode} =========")

    states, info = env.reset()
    state = states["coordinator"]

    good_epoch_apple = []
    good_epoch_dirt = []
    good_epoch_x1 = []
    # good_epoch_x2 = []
    good_epoch_x3 = []


    for step in tqdm(range(steps_per_epsiode)):
        num_cleaners, num_pickers = agentCoordinator.generate_roles()
        assignments = env.get_greedy_assignments(num_pickers, num_cleaners)
        actions = env.get_greedy_actions(assignments)
        next_states, rewards, dones, info = env.step(actions)
        next_state = next_states["coordinator"]
        reward = sum(rewards.values())

        agentCoordinator.step(state, reward, next_state)

        good_epoch_apple.append(info["apple"])
        good_epoch_dirt.append(info["dirt"])
        good_epoch_x1.append(info["cleaner"])
        good_epoch_x3.append(info["picker"])

        if episode > verbose_episode:
            print(f"========= Step {step} =========")
            print(f"info: {info}")

        state = next_state

        if dones["__all__"]:
            break

    ending_reward = info["total_apple_consumed"]

    print(f"ending reward: {ending_reward}")
    print(f"Current Epsilon: {agentCoordinator.epsilon}")
    print(f"========= End of Episode {episode} =========")

    ending_ep_rewards.append(ending_reward)
    print(ending_ep_rewards)

    if ending_reward >= 0:
        f_good.write(f"Epoch number: {episode}\n")
        f_good.write(f"Epoch reward: {ending_reward}\n")
        f_good.write(f"Epoch apple\n")
        f_good.write(f"{good_epoch_apple}\n\n")
        f_good.write(f"Epoch dirt\n")
        f_good.write(f"{good_epoch_dirt}\n\n")
        f_good.write(f"Epoch dirt cleaners\n")
        f_good.write(f"{good_epoch_x1}\n\n")
        f_good.write(f"Epoch apple pickers\n")
        f_good.write(f"{good_epoch_x3}\n\n")

    agentCoordinator.scheduler.step()

    # saving results
    model_saves.append((ending_reward, agentCoordinator.u_network.state_dict()))
    if len(model_saves) > 3:
        model_saves.popleft()

    if ending_reward > 2200 and episode > 10 and model_saves[0][0] > 2200:
        max_reward = ending_reward
        torch.save(model_saves[0][1], "model_save_oned")
        torch.save(model_saves[1][1], "model_save_oned1")


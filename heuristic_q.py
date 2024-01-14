from environments.one_d_cleanup_env_q import OneDCleanupEnvQ, CleanupRegion
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from agents.q_agent import QAgent
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda available: {torch.cuda.is_available()}")
np.set_printoptions(threshold=np.inf)

steps_per_episode = 1000
num_agents = 10
agent_ids = [str(i + 1) for i in range(num_agents)]
thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05
dirt_multiplier = 10

area = 150
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.1
lr = 0.0001
batch_size = 32
state_dim = 4
num_action_outputs = 1
action_size = 2
env = OneDCleanupEnvQ(agent_ids=agent_ids,
                     num_agents=num_agents,
                     area=area,
                     thresholdDepletion=thresholdDepletion,
                     thresholdRestoration=thresholdRestoration,
                     wasteSpawnProbability=wasteSpawnProbability,
                     appleRespawnProbability=appleRespawnProbability,
                     use_randomness=False,
                     dirt_multiplier=dirt_multiplier)
picker_q = QAgent(state_dim=state_dim,
                  action_size=action_size,
                  num_action_outputs=num_action_outputs,
                  gamma=gamma,
                  epsilon=epsilon,
                  epsilon_decay=epsilon_decay,
                  epsilon_min=epsilon_min,
                  lr=lr,
                  device=device,
                  batch_size=batch_size,
                  q_layers=[
                      (state_dim, 128),
                      (128, 64),
                      (64, action_size)
                  ])
cleaner_q = QAgent(state_dim=state_dim,
                   action_size=action_size,
                   num_action_outputs=num_action_outputs,
                   gamma=gamma,
                   epsilon=epsilon,
                   epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min,
                   lr=lr,
                   device=device,
                   batch_size=batch_size,
                   q_layers=[
                       (state_dim, 128),
                       (128, 64),
                       (64, action_size)
                   ])
ending_ep_rewards = []
num_episodes = 100
steps_per_epsiode = 1000
verbose_episode = num_episodes - 1

max_reward = 0
for episode in range(num_episodes):
    print(f"========= Episode {episode} =========")

    states, info = env.reset()

    #print(f"info: {info}")

    prev_assignments = {id: env._agents[id].region for id in agent_ids}

    picker_has_stepped = False
    cleaner_has_stepped = False

    for step in tqdm(range(steps_per_epsiode)):
        num_apples, num_dirt, _, _ = states["coordinator"]
        agent_frequency_in_dirt = num_dirt / (num_apples + num_dirt)
        num_cleaner = round(num_agents * agent_frequency_in_dirt)
        num_picker = num_agents - num_cleaner
        assignments = env.get_greedy_assignments(num_picker, num_cleaner)
        real_num_picker = len([id for id in agent_ids if assignments[id] == CleanupRegion.APPLE])
        real_num_cleaner = len([id for id in agent_ids if assignments[id] == CleanupRegion.WASTE])
        assert real_num_picker == num_picker
        assert real_num_cleaner == num_cleaner
        directions = {}
        simulated_actions = {}
        simulated_states = {}
        for agent_id in sorted(agent_ids):
            if simulated_actions:
                simulated_state = env.simulate_step(simulated_actions)[0]
            else:
                simulated_state = states
            simulated_states[agent_id] = simulated_state
            if assignments[agent_id] != prev_assignments[agent_id]:
                directions[agent_id] = 0
                simulated_actions[agent_id] = (assignments[agent_id], 0)
            elif assignments[agent_id] == CleanupRegion.APPLE:
                directions[agent_id] = -1 if (picker_q.act(np.array(simulated_state[agent_id])).flatten()[0] == 0) else 1
                simulated_actions[agent_id] = (CleanupRegion.APPLE, directions[agent_id])
            else:
                directions[agent_id] = -1 if (cleaner_q.act(np.array(simulated_state[agent_id])).flatten()[0] == 0) else 1
                simulated_actions[agent_id] = (CleanupRegion.WASTE, directions[agent_id])
        final_simulated_state = env.simulate_step(simulated_actions)[0]
        #print(f"actions: {actions}")
        actions = {agent_id: (assignments[agent_id], directions[agent_id]) for agent_id in agent_ids}
        next_states, rewards, dones, _, info = env.step(actions)

        # print('dirt rewards', {agent_id: rewards[agent_id] for agent_id in agent_ids if assignments[agent_id] == CleanupRegion.WASTE})
        # print('apple rewards', {agent_id: rewards[agent_id] for agent_id in agent_ids if assignments[agent_id] == CleanupRegion.APPLE})

        for agent_id in sorted(agent_ids):
            next_agent = int(agent_id) + 1
            if next_agent == num_agents + 1:
                next_simulated_state = final_simulated_state
            else:
                next_simulated_state = simulated_states[str(next_agent)]
            if assignments[agent_id] != prev_assignments[agent_id]:
                continue
            elif assignments[agent_id] == CleanupRegion.APPLE:
                action = 0 if actions[agent_id][1] == -1 else 1
                picker_q.step(np.array(simulated_states[agent_id][agent_id]), action, rewards[agent_id], np.array(next_simulated_state[agent_id]))
                picker_has_stepped = True
            else:
                action = 0 if actions[agent_id][1] == -1 else 1
                cleaner_q.step(np.array(simulated_states[agent_id][agent_id]), action, rewards[agent_id], np.array(next_simulated_state[agent_id]))
                cleaner_has_stepped = True

        if episode > verbose_episode:
            print(f"========= Step {step} =========")
            print(f"info: {info}")

        states = next_states

        prev_assignments = assignments

        if dones["__all__"]:
            break

    ending_reward = info["total_apple_consumed"]

    print(f"ending reward: {ending_reward}")
    print(f"========= End of Episode {episode} =========")
    print(f"Current Epsilon picker_q: {picker_q.epsilon}")
    print(f"Current Epsilon cleaner_q: {cleaner_q.epsilon}")

    ending_ep_rewards.append(ending_reward)
    print(ending_ep_rewards)
    # if picker_has_stepped:
    #     picker_q.scheduler.step()
    #     picker_has_stepped = False
    # if cleaner_has_stepped:
    #     cleaner_q.scheduler.step()
    #     cleaner_has_stepped = False

    if ending_reward > max_reward:
        max_reward = ending_reward

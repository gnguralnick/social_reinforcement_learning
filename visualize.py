import matplotlib.pyplot as plt

def plot_total_rewards(stats, fig=False):
    if fig:
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Total Rewards by Episode')
    total_by_episode = stats['total']
    plt.plot(total_by_episode)

def plot_average_rewards(stats, fig=False):
    if fig:
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Average Rewards by Episode')
    average_by_episode = stats['average']
    plt.plot(average_by_episode)

def plot_total_and_average_rewards(stats):
    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Total and Average Rewards by Episode')
    plot_total_rewards(stats)
    plot_average_rewards(stats)
    plt.legend(['Total Reward', 'Average Reward'])

def plot_agent_rewards(stats):
    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Rewards by Episode')
    for agent in stats:
        if agent == 'total' or agent == 'average':
            continue
        agent_rewards = [stats[agent][i]['total_reward'] for i in range(len(stats[agent]))]
        plt.plot(agent_rewards)

def plot_total_by_num_agents(stats_list, fig=False):
    if fig:
        plt.figure()
        plt.xlabel('Number of Agents')
        plt.ylabel('Total Reward')
        plt.title('Total Reward by Number of Agents')
    data = []
    for stats in stats_list:
        num_agents = len(stats) - 2 # -2 for 'total' and 'average'
        total_reward = stats['total'][-1]
        data.append((num_agents, total_reward))

    data.sort(key=lambda x: x[0])
    x, y = zip(*data)

    plt.plot(x, y)

def plot_average_by_num_agents(stats_list, fig=False):
    if fig:
        plt.figure()
        plt.xlabel('Number of Agents')
        plt.ylabel('Average Reward')
        plt.title('Average Reward by Number of Agents')
    data = []
    for stats in stats_list:
        num_agents = len(stats) - 2 # -2 for 'total' and 'average'
        average_reward = stats['average'][-1]
        data.append((num_agents, average_reward))

    data.sort(key=lambda x: x[0])
    x, y = zip(*data)

    plt.plot(x, y)

def plot_total_and_average_by_num_agents(stats_list):
    plt.figure()
    plt.xlabel('Number of Agents')
    plt.ylabel('Reward')
    plt.title('Total and Average Reward by Number of Agents')
    plot_total_by_num_agents(stats_list)
    plot_average_by_num_agents(stats_list)
    plt.legend(['Total Reward', 'Average Reward'])
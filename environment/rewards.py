import networkx as nx


class RewardScheme:
    def calculate_rewards(self, agent_rewards, step_info, env):
        raise NotImplementedError

    def get_terminated(self):
        termination_bonus = 50
        return termination_bonus


class Default(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        connected = step_info["connected"]
        collisions = step_info["collisions"]

        exploration_reward = 0.2
        disconnection_penalty = -0.5
        obstacle_penalty = -0.1
        timestep_penalty = -0.01

        for agent in env.agents:
            if collisions[agent]:
                agent_rewards[agent] += obstacle_penalty

            if not connected:
                agent_rewards[agent] += disconnection_penalty

            agent_loc = env.agent_locations[agent]
            if env.visited_tiles[agent_loc[0], agent_loc[1]] == 0:  # individual
                agent_rewards[agent] += exploration_reward

            agent_rewards[agent] += timestep_penalty


class Coverage(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        connected = step_info["connected"]
        collisions = step_info["collisions"]
        coverage = (step_info['coverage'] / 100)
        prev_coverage = (step_info['prev_coverage'] / 100)

        exploration_reward = (coverage - prev_coverage) * 100
        disconnection_penalty = -0.5
        obstacle_penalty = -0.1
        timestep_penalty = -0.01

        for agent in env.agents:
            if collisions[agent]:
                agent_rewards[agent] += obstacle_penalty

            if not connected:
                agent_rewards[agent] += disconnection_penalty

            agent_rewards[agent] += exploration_reward + timestep_penalty


class ExplorerMaintainer(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        obstacle_penalty = -1.0
        coverage_ratio = (step_info['coverage'] / 100)
        explorer_reward = 1.0 + (coverage_ratio ** 2)
        maintainer_reward = 0.5 * explorer_reward
        stagnation_penalty = -0.1
        disconnected = -2.0

        explorers = []
        maintainers = []

        for agent in env.agents:
            current_pos = env.agent_locations[agent]

            if step_info['collisions'][agent]:
                agent_rewards[agent] += obstacle_penalty

            if env.visited_tiles[current_pos[0], current_pos[1]] == 0:
                explorers.append(agent)
            else:
                maintainers.append(agent)

        if step_info['connected']:
            for agent in explorers:
                agent_rewards[agent] += explorer_reward

            if explorers:
                for agent in maintainers:
                    agent_rewards[agent] += maintainer_reward
            else:
                for agent in maintainers:
                    agent_rewards[agent] += stagnation_penalty

        else:
            for agent in env.agents:
                agent_rewards[agent] += disconnected


class Components(RewardScheme):
    # For Tile Discovery:
    #
    # Lone agent:    +0.5 + (0.0–0.5) + 0.2 - 3.2 = [-2.5, -2.0]
    # Team of 2/5:   +0.5 + (0.0–0.5) + 0.4 - 2.4 = [-1.5, -1.0]
    # Team of 3/5:   +0.5 + (0.0–0.5) + 0.6 - 1.6 = [-0.5, -0.0]
    # Team of 4/5:   +0.5 + (0.0–0.5) + 0.8 - 0.8 = [+0.5, +1.0]
    # Team of 5/5:   +0.5 + (0.0–0.5) + 1.0 - 0.0 = [+1.5, +2.0]

    # For No Discovery:
    #
    # Lone agent:     0.2 - 3.2 = -3.0
    # Team of 2/5:    0.4 - 2.4 = -2.0
    # Team of 3/5:    0.6 - 1.6 = -1.0
    # Team of 4/5:    0.8 - 0.8 =  0.0
    # Team of 5/5:    1.0 - 0.0 = +1.0
    def calculate_rewards(self, agent_rewards, step_info, env):
        total_teammate_bonus = 1.0
        missing_teammate_penalty = -0.8
        new_tile_bonus = 0.5
        total_coverage_bonus = 0.5
        coverage_ratio = (step_info['coverage'] / 100)
        coverage_bonus = (coverage_ratio ** 2) * total_coverage_bonus

        G: nx.Graph = step_info['graph']
        components = nx.connected_components(G)

        total_agents = env.num_agents
        for component in components:
            num_agents = len(component)
            teammate_bonus = (num_agents / total_agents) * total_teammate_bonus
            disconnect_penalty = (total_agents - num_agents) * missing_teammate_penalty

            for agent_idx in component:
                if f"agent_{agent_idx}" not in env.agents:  # base-station
                    continue

                # All agents get the disconnect penalty
                agent_rewards[f"agent_{agent_idx}"] += disconnect_penalty

                # Only discovering agents get discovery and teammate bonuses
                current_pos = env.agent_locations[agent_idx]
                if env.visited_tiles[current_pos[0], current_pos[1]] == 0:
                    agent_rewards[f"agent_{agent_idx}"] += new_tile_bonus + coverage_bonus + teammate_bonus
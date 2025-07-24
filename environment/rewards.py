import networkx as nx


class RewardScheme:
    def __init__(self, reward_scheme):
        self.reward_scheme = reward_scheme

    def calculate_rewards(self, agent_rewards, step_info, env):
        raise NotImplementedError


class Default(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        connected = step_info["connected"]
        collisions = step_info["collisions"]

        for i, agent in enumerate(env.agents):
            current_pos = env.agent_locations[i]

            if env.visited_tiles[current_pos[0], current_pos[1]] == 0 and connected:
                agent_rewards[agent] += self.reward_scheme.get('new_tile_visited_connected', 2.0)
            elif env.visited_tiles[current_pos[0], current_pos[1]] == 0:
                agent_rewards[agent] += self.reward_scheme.get('new_tile_visited_disconnected', -2.0)
            elif connected:
                agent_rewards[agent] += self.reward_scheme.get('old_tile_visited_connected', 2.0)
            else:
                agent_rewards[agent] += self.reward_scheme.get('old_tile_visited_disconnected', -2.0)

            if collisions[agent]:
                agent_rewards[agent] += self.reward_scheme.get('obstacle_penalty', -1.0)


class ExplorerMaintainer(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        explorers = []
        maintainers = []

        for i, agent in enumerate(env.agents):
            current_pos = env.agent_locations[i]

            if step_info['collisions'][agent]:
                agent_rewards[agent] += self.reward_scheme.get('obstacle_penalty', -0.1)

            if env.visited_tiles[current_pos[0], current_pos[1]] == 0:
                explorers.append(agent)
            else:
                maintainers.append(agent)

        if step_info['connected']:
            explorer_reward = self.reward_scheme.get('explorer', 1.0) + (step_info['coverage'] / 100)
            for agent in explorers:
                agent_rewards[agent] += explorer_reward

            if explorers:
                for agent in maintainers:
                    agent_rewards[agent] += self.reward_scheme.get('maintainer_percentage', 0.5) * explorer_reward
            else:
                for agent in maintainers:
                    agent_rewards[agent] += self.reward_scheme.get('stagnation_penalty', -0.1)

        else:
            for agent in env.agents:
                agent_rewards[agent] += self.reward_scheme.get('disconnected', -1.0)


class Components(RewardScheme):
    # For Tile Discovery:
    #
    # Lone agent: +0.5 + (0-0.5) + 0.2 - 3.2 = -2.5 to -2.0
    # Team of 2/5: +0.5 + (0-0.5) + 0.4 - 2.4 = -1.5 to -1.0
    # Team of 3/5: +0.5 + (0-0.5) + 0.6 - 1.6 = -0.5 to 0.0
    # Team of 4/5: +0.5 + (0-0.5) + 0.8 - 0.8 = +0.5 to +1.0
    # Team of 5/5: +0.5 + (0-0.5) + 1.0 - 0.0 = +1.5 to +2.0
    #
    # For No Discovery:
    #
    # Lone agent: -3.2
    # Team of 2/5: -2.4
    # Team of 3/5: -1.6
    # Team of 4/5: -0.8
    # Team of 5/5: 0.0
    def calculate_rewards(self, agent_rewards, step_info, env):
        total_teammate_bonus = self.reward_scheme.get('total_teammate_bonus', 1.0)
        missing_teammate_penalty = self.reward_scheme.get('missing_teammate_penalty', -0.8)
        total_coverage_bonus = self.reward_scheme.get('total_coverage_bonus', 0.5)
        new_tile_bonus = self.reward_scheme.get('new_tile', 0.5)

        G: nx.Graph = step_info['graph']
        components = nx.connected_components(G)

        total_agents = env.num_agents
        for component in components:
            num_agents = len(component)
            teammate_bonus = (num_agents / total_agents) * total_teammate_bonus
            disconnect_penalty = (total_agents - num_agents) * missing_teammate_penalty

            for agent_idx in component:
                if f"agent_{agent_idx}" not in env.agents: # base-station
                    break

                # All agents get the disconnect penalty
                agent_rewards[f"agent_{agent_idx}"] += disconnect_penalty

                # Only discovering agents get teammate bonus + discovery bonuses
                current_pos = env.agent_locations[agent_idx]
                if env.visited_tiles[current_pos[0], current_pos[1]] == 0:
                    coverage_bonus = (step_info['coverage'] / 100) * total_coverage_bonus
                    agent_rewards[f"agent_{agent_idx}"] += teammate_bonus + new_tile_bonus + coverage_bonus
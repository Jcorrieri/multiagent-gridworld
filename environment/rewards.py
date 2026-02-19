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

class PureCoverage(RewardScheme):
    def calculate_rewards(self, agent_rewards, step_info, env):
        collisions = step_info["collisions"] 

        exploration_reward = 1.0
        collision_penalty = -1.0
        revisit_penalty = 0.1

        for agent in env.agents:
            if collisions[agent]:
                agent_rewards[agent] += collision_penalty

            agent_loc = env.agent_locations[agent]
            if env.visited_tiles[agent_loc[0], agent_loc[1]] == 0:  # individual
                agent_rewards[agent] += exploration_reward
            else:
                agent_rewards[agent] += revisit_penalty

# class Coverage(RewardScheme):
#     def calculate_rewards(self, agent_rewards, step_info, env):
#             connected = step_info["connected"]
#             collisions = step_info["collisions"]
#             coverage = (step_info['coverage'] / 100)
#             prev_coverage = (step_info['prev_coverage'] / 100)

#             exploration_reward = max(0.0, coverage - prev_coverage) * 100 / env.num_agents
#             disconnection_penalty = -1.0 / env.num_agents
#             obstacle_penalty = -0.1
#             timestep_penalty = 0.00
            
#             for agent in env.agents:
#                 if collisions[agent]:
#                     agent_rewards[agent] += obstacle_penalty

#                 if not connected:
#                     agent_rewards[agent] += disconnection_penalty / env.num_agents

#                 agent_rewards[agent] += exploration_reward + timestep_penalty

#--- Used for v4 (LATEST GLOBAL MODEL)
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
    def calculate_rewards(self, agent_rewards, step_info, env):
        collisions = step_info["collisions"]
        coverage = (step_info['coverage'] / 100)
        prev_coverage = (step_info['prev_coverage'] / 100)

        missing_teammate_penalty = -0.8
        coverage_multiplier = 1.0 + (coverage ** 2)  # added coverage multiplier [1.0, 2.0]
        exploration_reward = (coverage - prev_coverage) * 100 * coverage_multiplier  # added coverage multiplier bonus
        obstacle_penalty = -0.1

        G: nx.Graph = step_info['graph']
        components = nx.connected_components(G)

        for component in components:
            num_agents = len(component)
            disconnect_penalty = (env.total_num_robots - num_agents) * missing_teammate_penalty

            for agent_idx in component:
                agent = f"agent_{agent_idx}"
                if agent not in env.agents:  # base-station
                    continue
                
                if collisions[agent]:
                    agent_rewards[agent] += obstacle_penalty

                agent_rewards[agent] += disconnect_penalty

                agent_rewards[agent] += exploration_reward

        for k in agent_rewards:
            agent_rewards[k] /= env.total_num_robots

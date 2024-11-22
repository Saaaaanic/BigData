import numpy as np


# Define the environment grid
class Grid:
    def __init__(self, grid, start, end, obstacles, penalty=-5, reward=10):
        self.grid = grid
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.penalty = penalty
        self.reward = reward
        self.actions = ['↑', '↓', '←', '→']  # up, down, left, right
        self.state_space = [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))]
        self.action_space = {action: idx for idx, action in enumerate(self.actions)}
        self.P = self._create_transition_probabilities()

    # transition do not have probabilities because u can't go from (0, 0) to (4, 4) in one turn,
    # so u will have 100% to go to sides, and 0% to every other state
    def _create_transition_probabilities(self):
        P = {}
        for state in self.state_space:
            P[state] = {}
            for action in self.actions:
                new_state = self._move(state, action)
                if new_state == self.end:
                    P[state][action] = (new_state, self.reward)  # Reward for reaching the goal
                elif new_state in self.obstacles:
                    P[state][action] = (new_state, self.penalty)  # Penalty for hitting obstacle
                else:
                    P[state][action] = (new_state, -0.1)  # Small penalty for moving
        return P

    def _move(self, state, action):
        i, j = state
        if action == '↑':
            return max(0, i - 1), j
        elif action == '↓':
            return min(len(self.grid) - 1, i + 1), j
        elif action == '←':
            return i, max(0, j - 1)
        elif action == '→':
            return i, min(len(self.grid[0]) - 1, j + 1)
        return state


# Policy iteration algorithm
def policy_iteration(env, gamma=0.9, theta=1e-4):
    policy = {state: np.random.choice(env.actions) for state in env.state_space if state not in env.obstacles}
    V = {state: 0 for state in env.state_space}

    def policy_evaluation():
        while True:
            delta = 0
            for state in env.state_space:
                if state == env.end:
                    continue
                v = V[state]
                action = policy.get(state, '↑')
                new_state, reward = env.P[state][action]
                V[state] = reward + gamma * V[new_state]
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break

    def policy_improvement():
        policy_stable = True
        for state in env.state_space:
            if state == env.end:
                continue
            old_action = policy.get(state, '↑')
            action_values = {}
            for action in env.actions:
                new_state, reward = env.P[state][action]
                action_values[action] = reward + gamma * V[new_state]
            best_action = max(action_values, key=action_values.get)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    while True:
        policy_evaluation()
        if policy_improvement():
            break

    return policy


# Test the algorithm
grid = np.zeros((5, 5))
start = (0, 0)
end = (4, 4)
obstacles = [(1, 0), (1, 1), (1, 2), (2, 1), (3, 3), (3, 4)]
env = Grid(grid, start, end, obstacles)

optimal_policy = policy_iteration(env)

# Print the optimal policy
print("Optimal Policy:")
for i in range(len(grid)):
    row = ""
    for j in range(len(grid[0])):
        state = (i, j)
        if state in obstacles:
            row += f" X({optimal_policy.get(state, ' ')}) "
        elif state == end:
            row += " F "
        else:
            row += f"  {optimal_policy.get(state, ' ')}   "
    print(row)
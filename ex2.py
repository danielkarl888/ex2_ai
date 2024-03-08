import time
import random
import pacman

id = ["318324563"]


def board_from_dicts(init_locations, init_pellets, N, M):
    board = [[10 for _ in range(M)] for _ in range(N)]
    for key in init_locations:
        if init_locations[key]:
            x, y = init_locations[key]
            board[x][y] = key * 10
    for x_pellet, y_pellet in init_pellets:
        board[x_pellet][y_pellet] += 1
    return board


def is_done_game():
    pass





class Controller:
    "This class is a controller for a Pacman game."

    def __init__(self, N, M, init_locations, init_pellets, steps):
        """Initialize controller for given game board and number of steps.
        This method MUST terminate within the specified timeout.
        N - board size along the coordinate y (number of rows)
        M - board size along the coordinate x (number of columns)
        init_locations - the locations of ghosts and Pacman in the initial state
        init_locations - the locations of pellets in the initial state
        steps - number of steps the controller will perform
        """

        start_time = time.time()
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration-exploitation tradeoff
        self.q_table = [[0.0] * 4 for i in range(M * N)]  # init the q_table
        self.steps = steps
        self.N = N
        self.M = M
        self.init_locations = init_locations
        self.init_pellets = init_pellets
        self.init_board = board_from_dicts(init_locations, init_pellets, N, M)
        self.game = pacman.Game(steps, self.init_board.copy())
        self.game.reset()
        moves = ['L', 'D', 'R', 'U']
        step = 0
        locations = self.init_locations
        pellets = self.init_pellets

        while step < steps:
            if time.time() - start_time > 5 :
                break
            current_state = self.N * locations[7][0] + locations[7][1]

            if random.random() < self.epsilon:  # Random action for exploration
                action = random.choice(moves)
            else:
                # Greedy action for exploitation
                # get the best action to make from current state according to the q-table
                a = max(range(4), key=lambda a: self.q_table[current_state][a])
                action = moves[a]

            next_state, reward = self.get_next_state_and_reward(action)
            self.q_table[current_state][action] += (self.alpha * (reward + self.gamma * max(self.q_table[next_state]) -
                                                                  self.q_table[current_state][action]))

            current_state = next_state
            if self.game.done:
                self.game.reset()

            step += 1


    def get_next_state_and_reward(self, action):
        p = 0.7
        reward = 0
        moves = list(self.game.actions.keys())
        if action not in moves:
            print("This is wrong!")

        if random.random() < p:
            reward += self.game.update_board(self.game.actions[action])
        else:
            moves.remove(action)
            reward += self.game.update_board(self.game.actions[random.choice(moves)])
        new_state = self.N * self.game.locations[7][0] + self.game.locations[7][1]
        return new_state, reward

    def choose_next_move(self, locations, pellets):
        "Choose next action for Pacman given the current state of the board."
        moves = ['L', 'D', 'R', 'U']
        state = self.N * locations[7][0] + locations[7][1]
        action = max(range(4), key=lambda a: self.q_table[state][a])
        return moves[action]
        # moves = ['L', 'D', 'R', 'U']
        # pacman_location = locations[7]
        # if random.random() < self.epsilon:
        #     action = random.choice(moves)
        #     return action
        # else:
        #     # Greedy action for exploitation
        #
        #     state = self.N * locations[7][0]
        #     # get the best action to make from current state according to the q-table
        #     action = max(range(4), key=lambda a: self.q_table[state][a])
        #     # get the next_state and reward
        #     next_state, reward = self.get_next_state_and_prize(locations, pellets)
        #     # update the q_table
        #     self.q_table[state][action] += (self.alpha * (reward + self.gamma * max(self.q_table[next_state]) -
        #                                                   self.q_table[state][action]))
        #
        #     return moves[action]

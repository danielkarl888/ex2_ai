import time
import random
import pacman

id = ["318324563"]


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

        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration-exploitation tradeoff
        self.q_table = [[0] * 4 for i in range(M * N)]  # init the q_table
        self.steps = steps
        self.N = N
        self.M = M

    def choose_next_move(self, locations, pellets):
        "Choose next action for Pacman given the current state of the board."
        moves = ['L', 'D', 'R', 'U']
        pacman_location = locations[7]
        if random.random() < self.epsilon:
            action = random.choice(moves)
            return action
        else:
            # Greedy action for exploitation
            action = max(range(4), key=lambda a: self.q_table[self.N * locations[7][0]][a])
            # update the q_table
            # self.q_table[self.N * locations[7][0]][action] += self.alpha * (reward + self.gamma)

            return moves[action]

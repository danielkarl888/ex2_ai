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
        self.alpha = 0.1        # Learning rate
        self.gamma = 0.9        # Discount factor
        self.epsilon = 0.1      # Exploration-exploitation tradeoff
        self.episodes = 4000    # number of episodes to make
        self.q_table = [[0.0] * 4 for i in range(M * N)]  # init the q_table with M*N rows and 4 cols
        self.steps = steps
        self.N = N
        self.M = M
        init_board = board_from_dicts(init_locations, init_pellets, N, M)
        # create a game object in order to simulate the game for the train
        self.game = pacman.Game(steps, init_board.copy())
        moves = ['L', 'D', 'R', 'U']
        for _ in range(self.episodes):
            if time.time() - start_time > 5:
                break
            step = 0
            # start a new game for each episode
            self.game.reset()
            while step < steps:
                if time.time() - start_time > 5:
                    break
                # get the locations from the game environment
                locations = self.game.locations
                # get the current state row location in the q-table
                current_state = self.N * locations[7][0] + locations[7][1]
                # choose action using epsilon greedy
                if random.random() < self.epsilon:  # Random action for exploration
                    a = random.randint(0, 3)
                else:
                    # Greedy action for exploitation
                    # get the best action to make from current state according to the q-table
                    a = max(range(4), key=lambda a: self.q_table[current_state][a])
                action = moves[a]
                # get the next state and the reward after making the action
                next_state, reward = self.get_next_state_and_reward(action)
                # Q table update equation after getting the next state and reward
                self.q_table[current_state][a] += (self.alpha * (reward + self.gamma * max(self.q_table[next_state]) -
                                                                 self.q_table[current_state][a]))
                # update the next state to be the current state for the next step
                current_state = next_state
                # check if the game is over
                if self.game.done:
                    # start a new game
                    self.game.reset()
                step += 1
        # print(self.q_table)

    def get_next_state_and_reward(self, action):
        p = 0.8
        reward = 0
        moves = list(self.game.actions.keys())
        if action not in moves:
            print("This is wrong!")
        # the game is stochastic, so we make the action only with some probability
        if random.random() < p:
            # get the reward from the game simulator and update the board and locations
            reward += self.game.update_board(self.game.actions[action])
        else:
            # make one of the other actions, chosen randomly
            moves.remove(action)
            reward += self.game.update_board(self.game.actions[random.choice(moves)])
        # update the new state
        new_state = self.N * self.game.locations[7][0] + self.game.locations[7][1]
        return new_state, reward

    def choose_next_move(self, locations, pellets):
        "Choose next action for Pacman given the current state of the board."
        moves = ['L', 'D', 'R', 'U']
        state = self.N * locations[7][0] + locations[7][1]
        # get the best action from the trained q-table
        action = max(range(4), key=lambda a: self.q_table[state][a])
        return moves[action]

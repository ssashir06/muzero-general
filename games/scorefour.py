import datetime
import os
import copy

import numpy
import torch
import random

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (5, 4, 16)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(64))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 8  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 64  # Maximum number of moves if game is not finished before
        self.num_simulations = 256  # Number of future moves self-simulated
        self.discount = 0.965  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 5  # Number of blocks in the ResNet
        self.channels = 30  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64, 64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64, 64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64, 64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 120  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0001942  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 64  # Number of game moves to keep for every batch element
        self.td_steps = 64  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = ScoreFour()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                x = int(
                    input(
                        f"Enter the X (1, 2, 3 or 4) to play for the player {self.to_play()}: "
                    )
                )
                y = int(
                    input(
                        f"Enter the Y (1, 2, 3 or 4) to play for the player {self.to_play()}: "
                    )
                )
                las = self.legal_actions()
                def f(act):
                    (_x, _y, _) = ScoreFour.convert_action_to_xyz(act)
                    return x-1 == _x and y-1 == _y

                match = [act for act in las if f(act)]
                if (len(match) == 1
                and x>=1 and x<=4
                and y>=1 and y<=4):
                    choice = match[0]
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        (x, y, z) = ScoreFour.convert_action_to_xyz(action_number)
        return f"Play X {x+1}, Y {y+1}, Z {z+1}"


class ScoreFour:
    def __init__(self):
        self.board = numpy.zeros((4, 4, 4), dtype="int32")
        self.player = random.choice([-1, 1])
        self.winlocs = []
        
        # create judgement values
        directions = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, -1),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, -1, -1),
        ]
        for (cx, cy, cz) in directions:
            ax = [0,1,2,3] if cx == 0 else [0] if cx == 1 else [3]
            ay = [0,1,2,3] if cy == 0 else [0] if cy == 1 else [3]
            az = [0,1,2,3] if cz == 0 else [0] if cz == 1 else [3]
            poss = numpy.array(numpy.meshgrid(ax, ay, az)).T.reshape(-1,3)
            for (bx, by, bz) in poss:
                self.winlocs.append(
                    [(bx + cx*i, by + cy*i, bz + cz*i) for i in range(0,4)]
                )

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((4, 4, 4), dtype="int32")
        self.player = random.choice([-1, 1])
        return self.get_observation()

    def step(self, action):
        (x, y, z) = ScoreFour.convert_action_to_xyz(action)
        self.board[x, y, z] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else self.bonus_point(action)

        self.player *= -1

        return self.get_observation(), reward, done

    def bonus_point(self, action):
        bonus = 0
        (ax, ay, az) = ScoreFour.convert_action_to_xyz(action)
        
        # Preventing enemy's win
        for i, poss in enumerate(self.winlocs):
            if any([(ax, ay, az) == p for p in poss]):
                if sum([1 for (px, py, pz) in poss if self.board[px, py, pz] == -self.player]) == 3:
                    bonus += 0.25
                    break
        
        # Scoreing items
        for i, poss in enumerate(self.winlocs):
            if all([(self.board[px, py, pz] in [0, self.player]) for (px, py, pz) in poss]):
                bonus += 0.0100
            if all([(self.board[px, py, pz] in [0, -self.player]) for (px, py, pz) in poss]):
                bonus -= 0.0025
        
        return bonus

    def get_observation(self):
        board_mine = numpy.where(self.board == self.player, 1, 0)
        board_enemy = numpy.where(self.board == -self.player, 1, 0)

        # legal actions
        legal = self.legal_actions()
        board_legal = numpy.zeros((4, 4, 4), dtype="int32")
        for i in legal:
            (px, py, pz) = ScoreFour.convert_action_to_xyz(i)
            board_legal[px, py, pz] = 1
        
        # blocks evaluations
        winloc_evaluate_mine = numpy.zeros((4, 4, 4), dtype="float32")
        winloc_evaluate_enemy = numpy.zeros((4, 4, 4), dtype="float32")
        for i, poss in enumerate(self.winlocs):
            count_mine = sum([1 for (px, py, pz) in poss if self.board[px, py, pz] == self.player])
            count_enemy = sum([1 for (px, py, pz) in poss if self.board[px, py, pz] == -self.player])
            for (px, py, pz) in poss:
                winloc_evaluate_mine[px, py, pz] += count_mine / 16
                winloc_evaluate_enemy[px, py, pz] += count_enemy / 16

        return [
            numpy.reshape(board_mine, (4, 16)),
            numpy.reshape(board_enemy, (4, 16)),
            numpy.reshape(board_legal, (4, 16)),
            numpy.reshape(winloc_evaluate_mine, (4, 16)),
            numpy.reshape(winloc_evaluate_enemy, (4, 16)),
        ]

    def legal_actions(self):
        legal = []
        for i in range(64):
            (x, y, z) = ScoreFour.convert_action_to_xyz(i)
            if self.board[x, y, z] == 0 and (z == 0 or self.board[x, y, z-1] != 0):
                legal.append(i)
        return legal
    
    def convert_action_to_xyz(action):
        return (
            action // 16,
            (action // 4) % 4,
            action % 4
        )
    
    def convert_xyz_to_action(x, y, z):
        return x * 16 + y * 4 + z

    def have_winner(self):
        for i, poss in enumerate(self.winlocs):
            count = sum([1 for (px, py, pz) in poss if self.board[px, py, pz] == self.player])
            if count == 4:
                return True
        return False

    def expert_action(self):
        las = self.legal_actions()

        # find winning point
        for action in las:
            if self.is_winning_action(action, self.player):
                return action

        # simulate enemy
        enemys = []
        for action in las:
            if self.is_winning_action(action, -self.player):
                enemys.append(action)
        if len(enemys) > 0:
            return random.choice(enemys)

        # find good position
        candidates = []
        point_max = None
        for action in las:
            point = self.get_position_importance(action)
            candidates.append([action, point])
            if point_max is None or point > point_max:
                point_max = point
        
        return random.choice(
            [ap[0] for ap in candidates if ap[1] == point_max]
        )

    def is_winning_action(self, action, player):
        board_copy = copy.deepcopy(self.board)

        # simulate enemy
        (x, y, z) = ScoreFour.convert_action_to_xyz(action)
        board_copy[x, y, z] = player

        for i, poss in enumerate(self.winlocs):
            count = sum([1 for (px, py, pz) in poss if board_copy[px, py, pz] == player])
            if count == 4:
                return True
        return False

    def get_position_importance(self, action):
        board_copy = copy.deepcopy(self.board)

        # simulate step
        (x, y, z) = ScoreFour.convert_action_to_xyz(action)
        board_copy[x, y, z] = self.player

        count_sum = 0
        for i, poss in enumerate(self.winlocs):
            count = (
                sum([1 for (px, py, pz) in poss if board_copy[px, py, pz] == self.player])
                - sum([1 for (px, py, pz) in poss if board_copy[px, py, pz] == -self.player])
            )
            count_sum += count ** 2

        return count_sum


    def render(self):
        print(self.board[::-1])

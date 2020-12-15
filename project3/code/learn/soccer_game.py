import numpy as np


def self_goal_reward(player_index):
    if player_index == 0:
        return np.array([100, -100])
    elif player_index == 1:
        return np.array([-100, 100])


def opponent_goal_reward(player_index):
    if player_index == 0:
        return np.array([-100, 100])
    elif player_index == 1:
        return np.array([100, -100])


# Soccer Game
class SoccerGame:
    def __init__(self):
        # top-left corner as (0,0)
        # bottom-right corner as (1,3)

        # player A is 1st element in position array
        # player B is 2nd element in position array

        # Initial position for A in the soccer grid is (0, 2)
        # Initial position for B in the soccer grid is (0, 1)
        self.position = [np.array([0, 2]), np.array([0, 1])]

        # goal column for player A is 0 while for player B is 3
        self.goal_column = [0, 3]

        # Initially, the ball is with player B, hence index is 1
        self.index_ball_holder = 1

    def grid_position(self, player_index):
        return self.position[player_index][0] * 4 + self.position[player_index][1]

    def move_player(self, player_index, movement):
        # rewards for player A and player B
        rewards = np.array([0, 0])
        done = 0

        # opponent index will be binary opposite of player index
        opponent_index = 1 - player_index

        # copy of current position
        new_position = self.position.copy()
        new_position[player_index] = self.position[player_index] + movement

        # player collides with opponent
        if (new_position[player_index] == self.position[opponent_index]).all():
            # if player has ball, the ball is lost to opponent
            if self.index_ball_holder == player_index:
                self.index_ball_holder = opponent_index

        # update position if no collision
        elif new_position[player_index][0] in range(0, 2) and new_position[player_index][1] in range(0, 4):
            self.position[player_index] = new_position[player_index]

            # scored for self
            if self.index_ball_holder == player_index and self.position[player_index][1] == self.goal_column[player_index]:
                rewards = self_goal_reward(player_index)
                done = 1

            # scored for opponent
            elif self.index_ball_holder == player_index and self.position[player_index][1] == self.goal_column[opponent_index]:
                rewards = opponent_goal_reward(player_index)
                done = 1
        return rewards, done

    def play_game(self, actions):
        # rewards for player A and player B
        default_rewards = np.array([0, 0])

        default_done = 0

        # {Index-Action} pair is as below:
        # 0 is North, 1 is East, 2 is South, 3 is West, 4 is Stick

        # Allowed movement based on action at the index
        allowed_movements = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]

        if actions[0] not in range(0, 5) or actions[1] not in range(0, 5):
            print('Action value is invalid')
        else:
            # moving the first player
            # random order of player movement
            # Player A is index 0, player B is index 1
            index_first_mover = np.random.randint(0, 2)
            rewards, done = self.move_player(index_first_mover, allowed_movements[actions[index_first_mover]])
            if done:
                return [self.grid_position(0), self.grid_position(1), self.index_ball_holder], rewards, done

            # moving the second player
            index_second_mover = 1 - index_first_mover
            rewards, done = self.move_player(index_second_mover, allowed_movements[actions[index_second_mover]])
            if done:
                return [self.grid_position(0), self.grid_position(1), self.index_ball_holder], rewards, done

        return [self.grid_position(0), self.grid_position(1), self.index_ball_holder], default_rewards, default_done

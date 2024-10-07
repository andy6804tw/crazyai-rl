import numpy as np
import pandas as pd
import time
import os

np.random.seed(2)  # reproducible

N_ROWS = 5   # the number of rows in the 2D world
N_COLS = 5   # the number of columns in the 2D world
ACTIONS = ['up', 'down', 'left', 'right']     # available actions
EPSILON = 0.9   # greedy policy
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move

TREASURE_POSITION = (N_ROWS-1, N_COLS-1)  # new treasure position at the bottom-right corner

# Add obstacles in the environment
OBSTACLES = [(1, 1), (2, 2), (3, 3)]  # obstacle positions


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action has no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()
    return action_name


def get_state_index(row, col):
    return row * N_COLS + col


def is_obstacle(row, col):
    return (row, col) in OBSTACLES


def get_env_feedback(S, A):
    row, col = divmod(S, N_COLS)

    if A == 'up':
        new_row, new_col = row - 1, col
    elif A == 'down':
        new_row, new_col = row + 1, col
    elif A == 'left':
        new_row, new_col = row, col - 1
    elif A == 'right':
        new_row, new_col = row, col + 1

    # Check if move is within bounds and not into an obstacle
    if 0 <= new_row < N_ROWS and 0 <= new_col < N_COLS and not is_obstacle(new_row, new_col):
        S_ = get_state_index(new_row, new_col)
    else:
        S_ = S  # hit wall or obstacle, stay in the same place

    # Reward handling
    if (new_row, new_col) == TREASURE_POSITION:
        S_ = 'terminal'
        R = 1
    else:
        R = 0

    return S_, R


def update_env(S, episode, step_counter):
    env_list = [['-'] * N_COLS for _ in range(N_ROWS)]
    treasure_row, treasure_col = TREASURE_POSITION
    env_list[treasure_row][treasure_col] = 'T'
    
    # Place obstacles
    for obstacle in OBSTACLES:
        env_list[obstacle[0]][obstacle[1]] = 'X'  # 'X' represents an obstacle

    if S == 'terminal':
        interaction = f'Episode {episode+1}: total_steps = {step_counter}'
        print(f'\r{interaction}', end='')
        time.sleep(2)
        print('\r                                 ', end='')
    else:
        row, col = divmod(S, N_COLS)
        env_list[row][col] = 'o'
        
        # Clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print the updated environment
        for row in env_list:
            print(' '.join(row))
        time.sleep(FRESH_TIME)


def rl():
    n_states = N_ROWS * N_COLS
    q_table = build_q_table(n_states, ACTIONS)
    
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0  # start at top-left corner
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update Q-table
            S = S_  # move to next state
            
            update_env(S, episode, step_counter+1)
            step_counter += 1

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\nQ-table:\n')
    print(q_table)

    
# https://github.com/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym
# https://gymnasium.farama.org/content/migration-guide/
# https://github.com/ProgramTraveler/DRL
# https://github.com/datawhalechina/joyrl-book/tree/main
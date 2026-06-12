import itertools
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PickAndPlaceEnvironment:
    """
        Problem 1: Pick-and-Place Robot
    """
    def __init__(self, logger=None, gamma=0.99, reward_task=100.0, 
                 reward_time=-1.0, reward_collision=-20.0, penalty_smooth=-5.0):
        self.logger = logger
        
        # if logger is provided, log the initialization parameters
        self.gamma = gamma
        self.reward_task = reward_task
        self.reward_time = reward_time
        self.reward_collision = reward_collision
        self.penalty_smooth = penalty_smooth
        
        # action space + gripper
        self.joint_actions = [-1, 0, 1]
        self.gripper_actions = ['open', 'keep', 'close']
        
        # action space 
        self.action_space = list(itertools.product(
            self.joint_actions, 
            self.joint_actions, 
            self.gripper_actions
        ))
        
        if self.logger:
            self.logger.info("PickAndPlaceEnvironment Initialized.")
            self.logger.info(f"Parameters: Gamma={self.gamma}, R_task={self.reward_task}, R_time={self.reward_time}")
            self.logger.info(f"Total actions available: {len(self.action_space)}")

    def calculate_reward(self, is_success, is_collision, dv1, dv2):
        """Reward: R = R_task + R_time + R_smooth + R_safety"""
        r_total = self.reward_time 
        
        if is_success:
            r_total += self.reward_task
        if is_collision:
            r_total += self.reward_collision
            
        r_smooth = self.penalty_smooth * ((dv1 ** 2) + (dv2 ** 2))
        r_total += r_smooth
        
        return r_total

    def dummy_step(self, action, mock_success=False, mock_collision=False):
        """ Agent takes an action, we return a reward based on the action and mock conditions """
        dv1, dv2, gripper = action
        reward = self.calculate_reward(mock_success, mock_collision, dv1, dv2)
        
        if self.logger:
            self.logger.info(f"Action Taken: dv1={dv1}, dv2={dv2}, gripper={gripper} | Reward: {reward}")
            
        return reward


class GridWorld2x2(gym.Env):
    """
    Problem 2: 2x2 Gridworld
    """
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        # State: 4 states (0=s1, 1=s2, 2=s3, 3=s4)
        self.observation_space = spaces.Discrete(4)
        # Action: 4 actions (0=up, 1=down, 2=left, 3=right)
        self.action_space = spaces.Discrete(4)
        
        #  Reward & Transition
        self.rewards = {0: 5, 1: 10, 2: 1, 3: 2}
        
        # P[state][action] = (probability, next_state, reward, terminated)
        ## 0=up, 1=down, 2=left, 3=right
        # (s' = s)
        self.P = {}
        for s in range(4):
            self.P[s] = {a: [] for a in range(4)}
            
        # s1 (index 0)
        self.P[0][0] = [(1.0, 0, self.rewards[0], False)] # up -> wall(0)
        self.P[0][1] = [(1.0, 2, self.rewards[0], False)] # down -> s3(2)
        self.P[0][2] = [(1.0, 0, self.rewards[0], False)] # left -> wall(0)
        self.P[0][3] = [(1.0, 1, self.rewards[0], False)] # right -> s2(1)

        # s2 (index 1)
        self.P[1][0] = [(1.0, 1, self.rewards[1], False)] # up -> wall(1)
        self.P[1][1] = [(1.0, 3, self.rewards[1], False)] # down -> s4(3)
        self.P[1][2] = [(1.0, 0, self.rewards[1], False)] # left -> s1(0)
        self.P[1][3] = [(1.0, 1, self.rewards[1], False)] # right -> wall(1)

        # s3 (index 2)
        self.P[2][0] = [(1.0, 0, self.rewards[2], False)] # up -> s1(0)
        self.P[2][1] = [(1.0, 2, self.rewards[2], False)] # down -> wall(2)
        self.P[2][2] = [(1.0, 2, self.rewards[2], False)] # left -> wall(2)
        self.P[2][3] = [(1.0, 3, self.rewards[2], False)] # right -> s4(3)

        # s4 (index 3)
        self.P[3][0] = [(1.0, 1, self.rewards[3], False)] # up -> s2(1)
        self.P[3][1] = [(1.0, 3, self.rewards[3], False)] # down -> wall(3)
        self.P[3][2] = [(1.0, 2, self.rewards[3], False)] # left -> s3(2)
        self.P[3][3] = [(1.0, 3, self.rewards[3], False)] # right -> wall(3)


class GridWorld5x5(gym.Env):
    """
    Problem 3 & 4: 5x5 Gridworld using Gymnasium interface.
    """
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        
        # 5x5 = 25 states (0 to 24)
        self.observation_space = spaces.Discrete(25)
        # 4 actions (0=Up, 1=Right, 2=Down, 3=Left)
        self.action_space = spaces.Discrete(4)
        
        self.grid_size = 5
        self.goal_state = 24 # (4, 4)
        # Grey states: (0,4)->4, (2,2)->12, (3,0)->15
        self.grey_states = [4, 12, 15]
        
        # Build Transition Model P
        self.P = {}
        for s in range(25):
            self.P[s] = {a: [] for a in range(4)}
            
            # If it's the goal state, the episode is over
            if s == self.goal_state:
                for a in range(4):
                    self.P[s][a] = [(1.0, s, 0, True)]
                continue
                
            row = s // self.grid_size
            col = s % self.grid_size
            
            for a in range(4):
                next_row, next_col = row, col
                if a == 0: next_row = max(0, row - 1)      # Up
                elif a == 1: next_col = min(4, col + 1)    # Right
                elif a == 2: next_row = min(4, row + 1)    # Down
                elif a == 3: next_col = max(0, col - 1)    # Left
                
                next_s = next_row * self.grid_size + next_col
                
                # Assign Rewards and Terminated flag
                terminated = False
                if next_s == self.goal_state:
                    reward = 10
                    terminated = True
                elif next_s in self.grey_states:
                    reward = -5
                else:
                    reward = -1
                    
                self.P[s][a] = [(1.0, next_s, reward, terminated)]
                
        if self.logger:
            self.logger.info("GridWorld5x5 Initialized.")
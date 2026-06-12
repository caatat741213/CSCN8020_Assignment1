
class GridWorld2x2:
    def __init__(self):
        self.states = ['s1', 's2', 's3', 's4']
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = {'s1': 5, 's2': 10, 's3': 1, 's4': 2}
        
        # 定義轉移邏輯 {state: {action: next_state}}
        # 依照你的手算邏輯，把撞牆的狀態設為原本的狀態
        self.transitions = {
            's1': {'up': 's1', 'down': 's3', 'left': 's1', 'right': 's2'},
            's2': {'up': 's2', 'down': 's4', 'left': 's1', 'right': 's2'},
            's3': {'up': 's1', 'down': 's3', 'left': 's3', 'right': 's4'},
            's4': {'up': 's2', 'down': 's4', 'left': 's3', 'right': 's4'}
        }

    def get_reward(self, state):
        return self.rewards[state]

    def get_next_state(self, state, action):
        return self.transitions[state][action]
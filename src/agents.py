
class ValueIterationAgent:
    def __init__(self, environment, gamma=0.9, logger=None):
        self.env = environment
        self.gamma = gamma
        self.logger = logger
        # 初始化 V0 = 0
        self.V = {s: 0.0 for s in self.env.states}
        
    def iterate(self, iterations=2):
        if self.logger:
            self.logger.info(f"Starting Value Iteration for {iterations} iterations.")
            
        for i in range(1, iterations + 1):
            new_V = {}
            for s in self.env.states:
                action_values = []
                # 針對每一個動作計算回報: R(s) + gamma * V(s')
                for a in self.env.actions:
                    next_s = self.env.get_next_state(s, a)
                    reward = self.env.get_reward(s)
                    val = reward + self.gamma * self.V[next_s]
                    action_values.append(val)
                
                # 取最大值
                new_V[s] = max(action_values)
                
            self.V = new_V # 更新 V
            if self.logger:
                self.logger.info(f"Iteration {i} Values: {self.V}")
                
        return self.V
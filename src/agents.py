import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4, logger=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.logger = logger
        # Initialize V(s) = 0
        self.V = np.zeros(self.env.observation_space.n)

    def iterate(self, num_iterations=None):
        #If num_iterations is provided, it runs exactly that many iterations.Otherwise, it runs until delta < theta.
        iteration_count = 0
        
        while True:
            delta = 0
            # Ensure synchronous updates (standard Value Iteration)
            new_V = np.copy(self.V)
            
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                action_values = []
                
                # Calculate expected return for all actions
                for a in range(self.env.action_space.n):
                    # In deterministic environments, list length is 1
                    prob, next_s, reward, terminated = self.env.P[s][a][0]
                    # Bellman optimality update logic
                    expected_return = reward if terminated else reward + self.gamma * self.V[next_s]
                    action_values.append(expected_return)
                
                new_V[s] = max(action_values)
                delta = max(delta, abs(v - new_V[s]))
            
            self.V = new_V
            iteration_count += 1
            
            if self.logger:
                self.logger.info(f"Iteration {iteration_count} Values: {self.V}")
                
            # Stop condition based on exact iterations (for Problem 2)
            if num_iterations is not None and iteration_count >= num_iterations:
                break
                
            # Stop condition based on convergence (for Problem 3)
            if num_iterations is None and delta < self.theta:
                if self.logger:
                    self.logger.info(f"Value Iteration converged after {iteration_count} iterations.")
                break
                
        return self.V, iteration_count
    

class InPlaceValueIterationAgent:
    # Problem 3 Variation: In-Place Value Iteration.
    def __init__(self, env, gamma=0.9, theta=1e-4, logger=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.logger = logger
        self.V = np.zeros(self.env.observation_space.n)
        
    def iterate(self, num_iterations=None):
        iteration_count = 0
        while True:
            delta = 0
            # note: this implementation updates V[s] immediately, which is the key difference from the standard Value Iteration.
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                action_values = []
                for a in range(self.env.action_space.n):
                    prob, next_s, reward, terminated = self.env.P[s][a][0]
                    expected_return = reward if terminated else reward + self.gamma * self.V[next_s]
                    action_values.append(expected_return)
                
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            iteration_count += 1
            
            if num_iterations is not None and iteration_count >= num_iterations:
                break
            if num_iterations is None and delta < self.theta:
                break
                
        return self.V, iteration_count
    

class OffPolicyMCAgent:
    """
    Problem 4: Off-policy MC control with Weighted Importance Sampling.
    Based on Sutton & Barto (2018) Figure 5.10, page 136.
    """
    def __init__(self, env, gamma=0.9, logger=None):
        self.env = env
        self.gamma = gamma
        self.logger = logger
        
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        
        # Initialize Q(s, a), C(s, a), and target policy pi
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.C = np.zeros((self.num_states, self.num_actions))
        # Greedy target policy (pi)
        self.target_policy = np.zeros(self.num_states, dtype=int)
        
    def generate_episode(self, max_steps=100):
        """Generate an episode using a random behavior policy."""
        episode = []
        # Random starting state (for better exploration in gridworlds)
        # Note: In standard gym, we usually use env.reset(), but for gridworlds 
        # starting anywhere helps MC coverage. Let's start from state 0 (top-left).
        s = 0 
        
        for step in range(max_steps):
            # Behavior policy mu: random action (probability = 1 / |A|)
            a = np.random.choice(self.num_actions)
            prob_mu = 1.0 / self.num_actions
            
            # Take action
            _, next_s, reward, terminated = self.env.P[s][a][0]
            
            episode.append((s, a, reward, prob_mu))
            
            if terminated:
                break
            s = next_s
            
        return episode
        
    def train(self, num_episodes=10000):
        if self.logger:
            self.logger.info(f"Starting Off-Policy MC training for {num_episodes} episodes...")
            
        for ep in range(num_episodes):
            episode = self.generate_episode()
            
            G = 0.0
            W = 1.0
            
            # Loop backwards through the episode
            for t in reversed(range(len(episode))):
                s_t, a_t, r_t_plus_1, prob_mu = episode[t]
                
                G = self.gamma * G + r_t_plus_1
                self.C[s_t, a_t] += W
                self.Q[s_t, a_t] += (W / self.C[s_t, a_t]) * (G - self.Q[s_t, a_t])
                
                # Update target policy to be greedy w.r.t Q
                self.target_policy[s_t] = np.argmax(self.Q[s_t])
                
                # If the action taken by behavior policy is NOT the greedy action,
                # pi(a|s) = 0, so W becomes 0 and we exit the loop.
                if a_t != self.target_policy[s_t]:
                    break
                    
                # Update W: W = W * (pi(a|s) / mu(a|s))
                # Since pi is deterministic greedy, pi(a|s) = 1.
                W = W * (1.0 / prob_mu)
                
        # Calculate final V(s) based on max Q(s,a)
        self.V = np.max(self.Q, axis=1)
        return self.V, self.target_policy
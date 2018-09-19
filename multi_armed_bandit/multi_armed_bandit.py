import time
import numpy as np


class Uniform_MAB:

    

    def __init__(self, n_arms=1, n_max=None, time_max=None):
        self.n_arms = None
        self.n_max = None
        self.time_max = None
        self.set(n_arms, n_max, time_max)


        
    def set(self, n_arms=None, n_max=None, time_max=None):
        if (n_arms is not None) and (n_arms > 0):
            self.n_arms = n_arms
        if (time_max is not None):
            self.time_max = time_max
        if (n_max is not None):
            self.n_max = n_max
        if (self.n_arms is None):
            self.n_arms = 1
        self.list_next = range(self.n_arms)
        self.n = 0
        self.time = time.time()
        self.list_rewards = [[] for i in range(self.n_arms)]
        self.mean_rewards = [0 for i in range(self.n_arms)]
        self.other_data = [[] for i in range(self.n_arms)]
        

        
    def next_arm(self):
        if (self.n_max is not None) and (self.n >= self.n_max):
            return None
        if (self.time_max is not None) and (time.time() - self.time >= self.time_max):
            return None
        return self._next_arm()


    
    def _next_arm(self):
        if len(self.list_next) == 0:
            self.list_next = range(self.n_arms)
        return self.list_next[0]

    

    def update_reward(self, reward, arm=None, other_data=None):
        if arm is None:
            arm = self._next_arm()
        self.list_next = self.list_next[1:]
        self.n += 1
        self.list_rewards[arm].append(reward)
        self.mean_rewards[arm] = np.mean(self.list_rewards[arm])
        self.other_data[arm].append(other_data)
        

import os
import time
import numpy as np
try:
    import signal
    class TimeoutException(Exception): pass
except ImportError:
    signal = None



class Uniform_MAB:
    """
    Basic multi armed bandit class, should be the parent of the other MAB classes
    It pulls the arms one after the other, in a uniform way (0,1,2,3,0,1,2,3,0,1,...)
    The list of rewards is in self.list_rewards and the mean of rewards is in self.mean_rewards
    The list of other data stored is in self.other_data
    """


    def __init__(self, n_arms=1, repeat_max=-1, n_max=-1, time_max=-1, arm_timeout=-1, timeout_reward=None, timeout_other_data="reset_default"):
        """
        n_arms : (int) = 1 : the number of arms
        repeat_max : (int) = -1 : stops when one arm is pulled more times than repeat_max
        n_max : (int) = -1 : stops when the total number of arms pulled is over n_max
        time_max : (float) = -1 : maximum time between setting this instance and the last arm pulled
        You can define these attributes later with the function set
        Before pulling arms, n_arms must be defined, and also either repeat_max or n_max or time_max
        """
        self.n_arms = None
        self.repeat_max = None
        self.n_max = None
        self.time_max = None
        self.arm_timeout = None
        self.timeout_reward = 0
        self.timeout_other_data = None
        self._handler = None
        self.set(n_arms, repeat_max, n_max, time_max, arm_timeout, timeout_reward, timeout_other_data)



    def set(self, n_arms=-1, repeat_max=-1, n_max=-1, time_max=-1, arm_timeout=-1, timeout_reward=None, timeout_other_data="reset_default"):
        """
        Set the inner parameters of this MAB class
        By default, it only adds contraints, set the arg to None to reset it
        n_arms : (int) = 1 : the number of arms
        repeat_max : (int) = -1 : stops when one arm is pulled more times than repeat_max
        n_max : (int) = -1 : stops when the total number of arms pulled is over n_max
        time_max : (float) = -1 : maximum time between setting this instance and the last arm pulled
        You can define these attributes later with the function set
        Before pulling arms, n_arms must be defined, and also either repeat_max or n_max or time_max
        """
        if (n_arms is None) or (n_arms > 0):
            self.n_arms = n_arms
        elif (n_arms != -1):
            print("MAB Warning : incorrect value for n_arms")
        if (repeat_max is None) or (repeat_max > 0):
            self.repeat_max = repeat_max
        elif (repeat_max != -1):
            print("MAB Warning : incorrect value for repeat_max")
        if (n_max is None) or (n_max > 0):
            self.n_max = n_max
        elif (n_max != -1):
            print("MAB Warning : incorrect value for n_max")
        if (time_max is None) or (time_max > 0):
            self.time_max = time_max
        elif (time_max != -1):
            print("MAB Warning : incorrect value for time_max")
        if (arm_timeout is None) or (arm_timeout > 0):
            if (signal is None):
                print("MAB Warning : cannot set arm_timeout, no module signal")
            elif (os.name != "posix"):
                print("MAB Warning : cannot set arm_timeout, it only works on linux")
            else:
                self.arm_timeout = arm_timeout
        elif (arm_timeout != -1):
            print("MAB Warning : incorrect value for arm_timeout")
        if (timeout_reward is None):
            self.timeout_reward = 0
        else:
            self.timeout_reward = timeout_reward
        if (timeout_other_data == "reset_default"):
            self.timeout_other_data = None
        else:
            self.timeout_other_data = timeout_other_data
        if (self.n_arms is None):
            self.n_arms = 1
        self.list_next = range(self.n_arms)
        self.n = 0
        self.time = time.time()
        self.list_rewards = [[] for i in range(self.n_arms)]
        self.mean_rewards = [0 for i in range(self.n_arms)]
        self.other_data = [[] for i in range(self.n_arms)]



    def next_arm(self):
        """
        Return the arm index of the next arm that should be pulled
        Return None if the maximum amount of arm pulled is reached
        Return None if the maximum amount of time pulling arms is reached
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if (self.n_max is not None) and (self.n >= self.n_max):
            return None
        if (self.time_max is not None) and (time.time() - self.time >= self.time_max):
            return None
        na = self._next_arm()
        if (self.repeat_max is not None) and (len(self.list_rewards[na]) >= self.repeat_max):
            return None
        else:
            return na




    def _next_arm(self):
        """
        You should use "next_arm" instead
        Return the arm index of the next arm that should be pulled
        It makes no verification concerning the maximum amount of arms pulled or maximum of time spent
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if len(self.list_next) == 0:
            self.list_next = range(self.n_arms)
        return self.list_next[0]



    def skip_arm(self):
        """
        Skip one arm from being pulled
        """
        self._next_arm()
        self.list_next = self.list_next[1:]



    def update_reward(self, reward, arm=None, other_data=None):
        """
        Update the reward obtained while pulling one arm
        reward : (float) : the reward obtained
        arm : (int) = None : the arm index pulled, default is self._next_arm()
        other_data : (*) = None : other data that we want to store per arm, ie like the reward
        """
        if arm is None:
            arm = self._next_arm()
        self.skip_arm()
        self.n += 1
        self.list_rewards[arm].append(reward)
        if (reward is None):
            self.mean_rewards[arm] = None
        elif (self.mean_rewards[arm] is not None):
            self.mean_rewards[arm] = np.mean(self.list_rewards[arm])
        self.other_data[arm].append(other_data)


    def _set_handler(self):
        if (self._handler is None):
            def _handler(signum, frame):
                raise TimeoutException
            self._handler = _handler
            signal.signal(signal.SIGALRM, self._handler)


    def init_timeout(self):
        if (self.arm_timeout is None):
            print("Warning : cannot set timeout")
        else:
            self._set_handler()
            signal.alarm(self.arm_timeout)


    def stop_timeout(self):
        if (self.arm_timeout is not None):
            signal.alarm(0)


    def pull(self, reward_function, *args, **kargs):
        """
        Pull one arm
        reward_function takes the index of the arm as first argument, and then *args and **kargs
        it returns only the reward
        """
        arm = self.next_arm()
        if arm is None:
            return False
        if (self.arm_timeout is None):
            reward = reward_function(arm, *args, **kargs)
            self.update_reward(reward, arm)
            return True
        else:
            self._set_handler()
            signal.alarm(self.arm_timeout)
            try:
                reward = reward_function(arm, *args, **kargs)
                signal.alarm(0)
            except TimeoutException:
                reward = self.timeout_reward
            self.update_reward(reward, arm)
            return True


    def pull_and_save_other_data(self, reward_function, *args, **kargs):
        """
        Pull one arm
        reward_function takes the index of the arm as first argument, and then *args and **kargs
        it returns (reward, other_data)
        """
        arm = self.next_arm()
        if arm is None:
            return False
        if (self.arm_timeout is None):
            reward, other_data = reward_function(arm, *args, **kargs)
            self.update_reward(reward, arm, other_data)
            return True
        else:
            self._set_handler()
            signal.alarm(self.arm_timeout)
            try:
                reward, other_data = reward_function(arm, *args, **kargs)
                signal.alarm(0)
            except TimeoutException:
                reward = self.timeout_reward
                other_data = self.timeout_other_data
            self.update_reward(reward, arm, other_data)
            return True
        

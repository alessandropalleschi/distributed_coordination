import numpy as np
import random

class SingleAgentMemory:
    def __init__(self, gamma, gae_lambda):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantage = []
        self.returns = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def store_memory(self, state, action, probs, vals, reward, done):
        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):

        self.compute_advantage()
        self.compute_empirical_return()

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            np.array(self.advantage),\
            np.array(self.returns)
    
    def compute_advantage(self):
        advantage = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae = 0

        # last_delta = self.rewards[-1] - self.vals[-1]
        # last_gae = last_delta + self.gamma * self.gae_lambda * (1 - int(self.dones[-1])) * last_gae
        # advantage[-1] = last_gae

        for t in reversed(range(len(self.rewards))):
            delta_t = self.rewards[t] + self.gamma * self.vals[t+1] * (1 - int(self.dones[t])) - self.vals[t]
            last_gae = delta_t + self.gamma * self.gae_lambda * (1 - int(self.dones[t])) * last_gae
            advantage[t] = last_gae
        
        self.advantage = advantage    

    def compute_empirical_return(self):
            # Calculate empirical returns outside the batches loop
            self.returns = np.zeros_like(self.advantage, dtype=np.float32)
            running_add = 0
            for t in reversed(range(len(self.returns))):
                running_add = running_add * self.gamma + self.rewards[t]
                self.returns[t] = running_add

    def store_last_value(self, value):
        self.vals.append(value)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.returns = []
        self.advantage = []

class PPOMemory:
    def __init__(self, batch_size, gamma, gae_lambda):
        self.memories = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def generate_batches(self):
        all_states, all_actions, all_probs, all_vals, all_rewards, all_dones, all_advantages, all_returns = [], [], [], [], [], [], [], []

        max_subset_size = min(len(self.memories), random.randint(1, len(self.memories)))
        selected_memories = random.sample(self.memories, max_subset_size)
        for memory in selected_memories:
            states, actions, probs, vals, rewards, dones, advantages, returns = memory.generate_batches()

            all_states.extend(states)
            all_actions.extend(actions)
            all_probs.extend(probs)
            all_vals.extend(vals)
            all_rewards.extend(rewards)
            all_dones.extend(dones)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        n_states = len(all_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(all_states), \
               np.array(all_actions), \
               np.array(all_probs), \
               np.array(all_vals), \
               np.array(all_rewards), \
               np.array(all_dones), \
               np.array(all_advantages),\
               np.array(all_returns),\
               batches
        

    def store_memory(self, state, action, probs, vals, reward, done, agent_id):
        if len(self.memories) <= agent_id:
            self.memories.extend([SingleAgentMemory(self.gamma, self.gae_lambda) for _ in range(agent_id - len(self.memories) + 1)])
        
        self.memories[agent_id].store_memory(state, action, probs, vals, reward, done)

    def store_last_value(self, value, agent_id):
        self.memories[agent_id].store_last_value(value)

    def clear_memory(self):
        for memory in self.memories:
            memory.clear_memory()



# import numpy as np


# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#         self.batch_size = batch_size

#     def generate_batches(self):
#         n_states = len(self.states)
#         batch_start = np.arange(0, n_states, self.batch_size)
#         indices = np.arange(n_states, dtype=np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i+self.batch_size] for i in batch_start]

#         return np.array(self.states),\
#             np.array(self.actions),\
#             np.array(self.probs),\
#             np.array(self.vals),\
#             np.array(self.rewards),\
#             np.array(self.dones),\
#             batches

#     def store_memory(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)

#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []

import numpy as np
EPS = 1e-6

class PrioritizedReplayBuffer():
    def __init__(self, 
                 max_samples=10000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992):
        self.max_samples = max_samples
        # self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=tuple)
        #------ by Komsun------
        self.ss_mem = np.empty(shape=(batch_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(batch_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(batch_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(batch_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(batch_size), dtype=np.ndarray)
        # ---------------------
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority
        # self.memory[self.next_index, 
        #             self.sample_index] = np.array(sample)
        # self.memory[self.next_index, 
        #             self.sample_index] = np.concatenate([np.array(x, dtype=np.float32).reshape(1) if np.isscalar(x) else x for x in sample])
        self.memory[self.next_index, 
                    self.sample_index] = sample
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries) + 1)
        else: # proportional
            priorities = entries[:, self.td_error_index] + EPS
        scaled_priorities = priorities**self.alpha        
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])
        
        for i in range(samples.shape[0]):
            pick = samples[i, 1] # pick is tuple of len 5
            self.ss_mem[i] = pick[0]
            self.as_mem[i] = pick[1]
            self.rs_mem[i] = pick[2]
            self.ps_mem[i] = pick[3]
            self.ds_mem[i] = pick[4]
        experiences = np.vstack(self.ss_mem), \
                      np.vstack(self.as_mem), \
                      np.vstack(self.rs_mem), \
                      np.vstack(self.ps_mem), \
                      np.vstack(self.ds_mem)

        # samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        samples_stacks = [np.vstack(batch_type.reshape(len(batch_type),1)) for batch_type in np.vstack(samples[:, self.sample_index].reshape(samples.shape[0],1)).T]
        
        samples_stacks = np.vstack(samples)
        
        
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, experiences

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])
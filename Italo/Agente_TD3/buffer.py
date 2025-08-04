import numpy as np
prioritized = True

import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, alpha=0.6, beta_start=1, beta_frames=100000):
        """
        max_size: tamanho máximo do buffer
        input_shape: shape do estado (ex: (dim,))
        n_actions: número de ações (dimensão da ação)
        alpha: expoente que controla a força da priorização (0 = sem priorização)
        beta_start: valor inicial de beta para importance sampling
        beta_frames: número de frames para annealing de beta até 1.0
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.priorities = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward.item() if hasattr(reward, 'item') else reward
        self.terminal_memory[index] = 1 - done  # 1 se não acabou, 0 se acabou

        # Prioridade máxima para garantir que amostra será possível
        max_prio = self.priorities.max() if self.mem_cntr > 0 else 1.0
        self.priorities[index] = max_prio

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, frame_idx=None):
        max_mem = min(self.mem_cntr, self.mem_size)
        prios = self.priorities[:max_mem] ** self.alpha
        total = prios.sum()
        probs = prios / total

        indices = np.random.choice(max_mem, batch_size, p=probs)

        # Annealing beta de beta_start até 1.0 ao longo de beta_frames steps
        if frame_idx is None:
            beta = self.beta_start
        else:
            beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * frame_idx / self.beta_frames)

        weights = (max_mem * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normaliza para que maior peso seja 1

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        terminal = self.terminal_memory[indices]

        return states, actions, rewards, states_, terminal, indices, weights

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        # Atualiza as prioridades vetorialmente
        self.priorities[indices] = np.abs(td_errors) + epsilon


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
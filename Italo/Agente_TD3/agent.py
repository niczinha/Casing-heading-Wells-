import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agente_TD3.buffer import PrioritizedReplayBuffer
from env import DualWellEnv
env = DualWellEnv() # Definindo o ambiente

num_obs = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_obs))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims + n_actions, 250)
        self.fc2 = nn.Linear(250, 250)
        self.q = nn.Linear(250, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, 100)
        self.fc2 = nn.Linear(100, 100)
        self.mu = nn.Linear(100, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return T.tanh(self.mu(x))

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2,
                max_size=1_000_000, batch_size=256, policy_noise=0.2, noise_clip=0.5,
                policy_delay=2):
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.learn_step = 0
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(alpha, input_dims, n_actions, name='actor')
        self.actor_target = ActorNetwork(alpha, input_dims, n_actions, name='actor_target')
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_2')
        self.critic_target_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_target_1')
        self.critic_target_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_target_2')
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.memory = PrioritizedReplayBuffer(max_size, input_dims, n_actions, alpha=0.6)

    def choose_action(self, observation, noise_scale=0.2):
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with T.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if noise_scale > 0:
            action += noise_scale * np.random.normal(size=action.shape)
        ub = np.array(upper_bound)
        lb = np.array(lower_bound)
        # Escala de [-1, 1] para [lb, ub]
        action = np.clip(0.5 * (action + 1) * (ub - lb) + lb, lb, ub)
        
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done, idxs, is_weights = \
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float32).to(self.device)
        action = T.tensor(action, dtype=T.float32).to(self.device)
        reward = T.tensor(reward, dtype=T.float32).unsqueeze(1).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float32).to(self.device)
        done = T.tensor(done, dtype=T.float32).unsqueeze(1).to(self.device)
        is_weights = T.tensor(is_weights, dtype=T.float32).unsqueeze(1).to(self.device)

        with T.no_grad():
            noise = (T.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(new_state)
            next_action = (next_action + noise).clamp(-1, 1)

            q1_next = self.critic_target_1(new_state, next_action)
            q2_next = self.critic_target_2(new_state, next_action)
            q_target = reward + self.gamma * (1 - done) * T.min(q1_next, q2_next)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        td_errors = (q1 - q_target).detach()

        # Loss ponderado por importance sampling
        critic_1_loss = (is_weights * (q1 - q_target).pow(2)).mean()
        critic_2_loss = (is_weights * (q2 - q_target).pow(2)).mean()

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        # Atualiza prioridades no buffer
        self.memory.update_priorities(idxs, td_errors.cpu().numpy().flatten())

        # Atualiza ator e redes-alvo
        if self.learn_step % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

        self.learn_step += 1


    def update_network_parameters(self):
        for target, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

        for t_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

        for t_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target_1.save_checkpoint()
        self.critic_target_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target_1.load_checkpoint()
        self.critic_target_2.load_checkpoint()

    # Novos métodos para controle de modo de treino / avaliação
    def set_eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.critic_target_1.eval()
        self.critic_target_2.eval()

    def set_train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic_1.train()
        self.critic_2.train()
        self.critic_target_1.train()
        self.critic_target_2.train()


if __name__ == "__main__":
    import torch as T
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import os

    # Parâmetros de exemplo
    input_dims = 4
    n_actions = 2
    device = 'cuda' if T.cuda.is_available() else 'cpu'

    # Estado fictício
    state = T.tensor(np.random.randn(1, input_dims), dtype=T.float32).to(device)

    # Criar e testar o ator
    actor = ActorNetwork(alpha=0.005, input_dims=input_dims, n_actions=n_actions, name='td3_actor_test').to(device)
    actor.eval()  # modo avaliação

    with T.no_grad():
        action = actor(state)

    print(f"Ação (tanh): {action.cpu().numpy()}")

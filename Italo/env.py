import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
import casadi as ca
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Simulation_wells import fun, modelo


class RiserModel:
    def __init__(self, p, m, steps, dt):
        self.p = p
        self.m = m
        self.steps = steps
        self.dt = dt
        self.gor1 = 0.08
        self.gor2 = 0.10

    def fun(self, t, x, par):
        return fun(t, x, par)

    def createModelo(self):
        def model_func(x, par, gor):
            return modelo(x, par)
        return model_func


class DualWellEnv(Env):
    def __init__(self):
        super(DualWellEnv, self).__init__()

        self.dt = 1.0
        self.sim = RiserModel(p=10, m=2, steps=100000, dt=self.dt)
        self.f_modelo = self.sim.createModelo()

        self.obs_min = np.array([9e6, 9e6, 10, 10], dtype=np.float32)
        self.obs_max = np.array([15e6, 15e6, 20, 20], dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.action_real_low = np.array([0, 0], dtype=np.float32)
        self.action_real_high = np.array([10, 10], dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.initial_state = np.array([4986.94900064, 4046.70069931,  659.3209184,   615.94059295, 1501.95016371,
 1778.69356488,  134.76380012,  344.00375514], dtype=np.float64)
        self.x = self.initial_state.copy()
        self.time = 0

        self.u1_History = []
        self.u2_History = []
        self.history = []

        
        self.acumulado_o1 = 0
        self.acumulado_o2 = 0
        self.base_gor1, self.base_gor2 = 0.041, 0.051
        self.max_steps = 2 * 60 * 60  # 10 horas em segundos

    def normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def denormalize_action(self, action_norm):
        return 0.5 * (action_norm + 1) * (self.action_real_high - self.action_real_low) + self.action_real_low

    def atualizar_gor(self):
        fator1 = 1e-8 * 0
        fator2 = 2e-8 * 0
        gor1 = self.base_gor1 + fator1 * self.acumulado_o1
        gor2 = self.base_gor2 + fator2 * self.acumulado_o2
        gor1 = min(gor1, 0.125)
        gor2 = min(gor2, 0.125)
        self.gor = (gor1, gor2)
        return self.gor

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.initial_state.copy()
        self.time = 0
        self.history = []
        self.gor = (0.05, 0.60)
        self.u1_History = []
        self.u2_History = []
        self.acumulado_o1 = 0
        self.acumulado_o2 = 0
        self.base_gor1, self.base_gor2 = np.random.uniform(0.07, 0.09), np.random.uniform(0.05, 0.09)

        par = np.array([1.0, 1.0, 1, 1, 0.02, 0.02])
        self.outputs = self.f_modelo(self.x, par, self.gor)
        self.history.append(np.array(list(self.outputs.values()), dtype=np.float32))

        obs = np.array([
            self.outputs['Pbh1'],
            self.outputs['Pbh2'],
            self.outputs['wpo1'],
            self.outputs['wpo2']

        ], dtype=np.float32)
        obs_norm = self.normalize_obs(obs)
        return obs_norm, {}

    def reward(self, action_real):
        preco_oleo = 100.0
        custo_gas = 7.0
        wto = self.outputs['wto_riser']  # produção de óleo total [kg/s]
        receita = preco_oleo * wto
        custo = custo_gas * (action_real[0] + action_real[1])
        lucro = receita - custo
        reward = lucro / 100000
        reward = 500*(reward - 0.03405)
        #print(reward)
        return float(reward)

    def step(self, action_norm, study=False, eval=False):
        #print(f"An",action_norm)
        action_real = self.denormalize_action(action_norm)
        #print(f"Ar",action_real)
        if study:
            action_real = 4,5
        self.u1_History.append(action_real[0])
        self.u2_History.append(action_real[1])

        self.atualizar_gor()

        par = np.array([action_real[0], action_real[1], 1, 1, self.gor[0], self.gor[1]], dtype=np.float64)

        dx = self.sim.fun(0, self.x, par) 
        dx_np = dx.full().flatten()
        x_next = self.x + dx_np * self.dt

        self.outputs = self.f_modelo(x_next, par, self.gor)
        self.x = x_next

        self.history.append(np.array(list(self.outputs.values()), dtype=np.float32))

        obs = np.array([
            self.outputs['Pbh1'],
            self.outputs['Pbh2'],
            self.outputs['wpo1'],
            self.outputs['wpo2']
        ], dtype=np.float32)

        obs_norm = self.normalize_obs(obs)
        reward = self.reward(action_real)

        self.acumulado_o1 += float(self.outputs['wpo1']) * self.dt
        self.acumulado_o2 += float(self.outputs['wpo2']) * self.dt

        self.time += self.dt
        done = self.time >= self.max_steps

        return obs_norm, reward, done, False, {}

    def render(self, save_fig=True, show_fig=True, fig=None):
        if len(self.history) == 0:
            print("Nenhum dado para renderizar. Execute steps antes.")
            return None

        saidas = np.array(self.history).T
        keys = list(self.outputs.keys())
        saidas_dict = {k: saidas[i] for i, k in enumerate(keys)}

        tempo = np.linspace(0, self.dt * len(self.history), len(self.history)) / 3600
        tempo_u = np.linspace(0, self.dt * len(self.u1_History), len(self.u1_History)) / 3600

        t_min = 0
        mask = tempo >= t_min
        mask_u = tempo_u >= t_min

        tempo = tempo[mask]
        tempo_u = tempo_u[mask_u]

        u1 = np.array(self.u1_History)[mask_u]
        u2 = np.array(self.u2_History)[mask_u]

        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        else:
            fig.clf()

        gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :])]

        axs[0].plot(tempo, saidas_dict['wpo1'][mask], label='Óleo Poço 1', color='b')
        axs[0].set_title('Poço 1 - Vazões Produzidas')
        axs[0].legend()
        axs[0].set_ylabel('Vazão [kg/s]')
        axs[0].set_xlabel('Tempo [h]')
        axs[0].grid(True)

        axs[1].plot(tempo, saidas_dict['wpo2'][mask], label='Óleo Poço 2', color='b')
        axs[1].set_title('Poço 2 - Vazões Produzidas')
        axs[1].legend()
        axs[1].set_ylabel('Vazão [kg/s]')
        axs[1].set_xlabel('Tempo [h]')
        axs[1].grid(True)

        axs[2].plot(tempo_u, u1, label='Gás Injetado Poço 1', color='k')
        axs[2].set_title('Poço 1 - Vazão de Gás Injetado')
        axs[2].set_ylabel('Vazão [kg/s]')
        axs[2].set_xlabel('Tempo [h]')
        axs[2].grid(True)

        axs[3].plot(tempo_u, u2, label='Gás Injetado Poço 2', color='k')
        axs[3].set_title('Poço 2 - Vazão de Gás Injetado')
        axs[3].set_ylabel('Vazão [kg/s]')
        axs[3].set_xlabel('Tempo [h]')
        axs[3].grid(True)

        axs[4].plot(tempo, saidas_dict['Pbh1'][mask], label='Pbh1', color='blue')
        axs[4].plot(tempo, saidas_dict['Pbh2'][mask], label='Pbh2', color='black')
        axs[4].set_title('Pressão no Fundo dos Poços')
        axs[4].set_xlabel('Tempo [h]')
        axs[4].set_ylabel('Pressão [Pa]')
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_fig:
            os.makedirs('img', exist_ok=True)
            plt.savefig('img/DualWellEnv_latest.png', dpi=300, bbox_inches='tight')

        if show_fig:
            plt.draw()
            plt.pause(0.001)
        print(f"wpo1",saidas_dict['wpo1'][-1],"wpo2",saidas_dict['wpo2'][-1])

        #print(f"Estado",self.x)

        return fig



if __name__ == "__main__":
    env = DualWellEnv()
    print("Tamanho do espaço de observação:", env.observation_space.shape[0])
    print("Tamanho do espaço de ação:", env.action_space.shape[0])
    print("Máximo valor da ação:", env.action_space.high)
    print("Mínimo valor da ação:", env.action_space.low)

    obs, _ = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Recompensa total do episódio: {total_reward}")
    env.render()

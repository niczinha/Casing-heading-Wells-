import os
import numpy as np
from env import DualWellEnv
from Agente_TD3.agent import TD3Agent

# Configurações do agente (devem bater com as usadas no treinamento)
config = {
    "alpha": 1e-4,
    "beta": 5e-4,
    "tau": 0.005,
    "gamma": 0.99,
    "batch_size": 256,
    "buffer_size": 350000,
    "n_actions": 2,
    "save_dir": "results_td3",
}

def evaluate_policy(agent, env, episodes=5, render=False):
    rewards = []
    fig = None
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = agent.choose_action(obs, noise_scale=0)  # ação sem ruído
            if steps < 60*60:
                action = np.array([-0.9,-0.9])
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        print(f"[Eval] Episódio {ep+1}: Recompensa = {total_reward:.2f}")
        if render:
            fig = env.render(show_fig=False, save_fig=True, fig=fig)
    mean_reward = np.mean(rewards)
    print(f"\n[Eval] Recompensa média após {episodes} episódios: {mean_reward:.2f}")
    return mean_reward

def main():
    env = DualWellEnv()
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    agent = TD3Agent(alpha=config["alpha"],
                     beta=config["beta"],
                     input_dims=num_obs,
                     tau=config["tau"],
                     env=env,
                     gamma=config["gamma"],
                     n_actions=num_actions,
                     max_size=config["buffer_size"],
                     batch_size=config["batch_size"])

    agent.load_models()
    evaluate_policy(agent, env, episodes=5, render=True)

if __name__ == "__main__":
    main()

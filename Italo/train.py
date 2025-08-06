import numpy as np
import os
from env import DualWellEnv
from Agente_TD3.agent import TD3Agent
# Configurações específicas para TD3
config = {
    "alpha": 1e-4,          # menor lr para ator
    "beta": 5e-4,           # menor lr para críticos
    "tau": 0.005,
    "gamma": 0.99,
    "batch_size": 512,
    "buffer_size": 500000,
    "n_actions": 2,
    "n_episodes": 300,
    "max_steps": 10*60*60,
    "render_every": 10,
    "save_dir": "results_td3",
    "initial_noise": 0.01,
    "final_noise": 0.001,
    "noise_decay_episodes": 100,
    "policy_noise": 0.1,
    "noise_clip": 0.3,
    "policy_delay": 2
}

def evaluate_policy(agent, env, episodes=1):
    total_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(obs, noise_scale=0)  # sem ruído
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def train_td3():
    env = DualWellEnv() # Definindo o ambiente

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
                     batch_size=config["batch_size"],
                     policy_noise=config["policy_noise"],
                     noise_clip=config["noise_clip"],
                     policy_delay=config["policy_delay"])

    n_episodes = config["n_episodes"]
    max_steps = config["max_steps"]
    render_every = config["render_every"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    rewards = []
    fig = None
    agent.load_models()
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # Decaimento linear do ruído de exploração
        noise_scale = max(
            config["final_noise"],
            config["initial_noise"] - (config["initial_noise"] - config["final_noise"]) * (episode / config["noise_decay_episodes"])
        )

        while not done and steps < max_steps:
            
            if steps % 20 == 0:
                action = agent.choose_action(obs, noise_scale=noise_scale)


            #if steps < 60*60:
            #    action = np.array([-0.9,-0.9])


            #print(f"Episode {episode}, Step {steps}, Action: {action}, Noise Scale: {noise_scale}")
            next_obs, reward, done, _, _ = env.step(action)
            #print(f"Step {steps}: Action: {action}, Reward: {reward}, Done: {done}")
            agent.remember(obs, action, reward, next_obs, done)

            # Aprendizado e captura das perdas
            if steps % (60) == 0:
                losses = agent.learn()
            obs = next_obs
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-10:])

        print(f"[TD3] Ep {episode} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Noise: {noise_scale:.3f}")

        # Se retornou perdas, mostra elas
        if losses is not None:
            critic_1_loss, critic_2_loss, actor_loss = losses
            if actor_loss is not None:
                print(f"Losses | Critic1: {critic_1_loss:.8f}, Critic2: {critic_2_loss:.8f}, Actor: {actor_loss:.8f}")
        if episode % 1 == 0:
            fig = env.render(show_fig=False, save_fig=True, fig=fig)
        if (episode + 1) % 20 == 0:
            print("Saving Models")
            agent.save_models()

        if (episode+1) % 25 == 0:
            # Avaliação periódica sem ruído
            eval_reward = evaluate_policy(agent, env, episodes=2)
            print(f"[TD3 Eval] Ep {episode} | Avg Eval Reward: {eval_reward:.2f}")

            fig = env.render(show_fig=False, save_fig=True, fig=fig)

    np.save(os.path.join(save_dir, "rewards_td3.npy"), np.array(rewards))
    print("[TD3] Treinamento finalizado.")

if __name__ == "__main__":
    train_td3()
   
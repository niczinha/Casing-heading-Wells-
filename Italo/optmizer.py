import casadi as ca
import numpy as np
from env import DualWellEnv  # ou use o caminho adequado se estiver em outro diretório


def optimize_constant_action(horizon=3600, dt=1.0):
    env = DualWellEnv()
    num_steps = int(horizon / dt)

    u = ca.MX.sym("u", 2)  # ação constante

    x = ca.MX(env.initial_state)
    acumulado_o1 = 0
    acumulado_o2 = 0
    gor1_base = env.base_gor1
    gor2_base = env.base_gor2
    total_reward = 0

    for _ in range(num_steps):
        gor1 = ca.fmin(gor1_base + 1e-8 * acumulado_o1*0, 0.125)
        gor2 = ca.fmin(gor2_base + 2e-8 * acumulado_o2*0, 0.125)
        par = ca.vertcat(u[0], u[1], 1, 1, gor1, gor2)

        dx = env.sim.fun(0, x, par)
        x = x + dx * dt

        outputs = env.f_modelo(x, par, (gor1, gor2))
        wto = outputs['wto_riser']
        custo = 7.0 * (u[0] + u[1])
        receita = 100.0 * wto
        lucro = receita - custo
        reward = 10 * ((lucro / 100000) - 0.033)
        total_reward += reward

        acumulado_o1 += outputs['wpo1'] * dt
        acumulado_o2 += outputs['wpo2'] * dt

    J = -total_reward  # minimiza custo negativo
    nlp = {'x': u, 'f': J}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {
    'ipopt.print_level': 5,     # nível maior mostra mais detalhes
    'ipopt.sb': 'yes',          # mostra status na saída padrão
    'print_time': True
})

    lbx = [0.0, 0.0]
    ubx = [5.0, 5.0]
    sol = solver(x0=[2.0, 2.0], lbx=lbx, ubx=ubx)

    u_opt = sol['x'].full().flatten()
    print(f"Ação ótima constante encontrada: {u_opt}")
    return u_opt




if __name__ == "__main__":
    best_action = optimize_constant_action(horizon=3600*5)

    env = DualWellEnv()
    obs, _ = env.reset()
    total_reward = 0

    while True:
        obs, reward, done, _, _ = env.step(best_action, study=True)
        total_reward += reward
        if done:
            break

    print(f"Recompensa total com ação ótima constante: {total_reward}")
    env.render()

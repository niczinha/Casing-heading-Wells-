import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Carrega o array do arquivo
data = np.load('results_td3/rewards_td3.npy')

# Exibe o conteúdo (ou parte dele)
print(data)
print(data.shape)
print(data.dtype)
import numpy as np
import matplotlib.pyplot as plt

# Carrega os dados


# Plota os dados
plt.plot(data)
plt.title("Recompensas por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()

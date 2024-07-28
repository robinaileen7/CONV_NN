import numpy as np
from scipy.stats import bernoulli

time_step = 252
N = 10000

np.random.seed(10)
N_asset = 6
portfolio_size = 5
N_sigma = 2
n_param = N_sigma + 2
#miu_asset = np.random.uniform(0.05, 0.1, N_asset)
risk_free = 0.05

sigma_1_asset = np.random.uniform(0.15, 0.4, N_asset)
sigma_2_asset = np.random.uniform(0.15, 0.4, N_asset)
sigma_asset = [sigma_1_asset, sigma_2_asset]

# Define market price of risk to extract miu; to prevent arbitrage-free
lambda_asset = np.random.uniform(0.01, 0.03, N_sigma)

miu_asset = np.array(np.sum([sigma_asset[i]*lambda_asset[i] for i in range(N_sigma)], axis=0)+risk_free)

# J_apt_1_asset = np.random.uniform(-0.05, -0.1, N_asset)
# J_apt_2_asset = np.random.uniform(0, 0.05, N_asset)
J_apt_1_asset = np.array([np.random.uniform(x-0.15, 0) for x in miu_asset])
J_apt_2_asset = np.array([np.random.uniform(x+0.02, 0) for x in miu_asset])

J_apt_asset = [J_apt_1_asset, J_apt_2_asset]

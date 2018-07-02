import matplotlib.pyplot as plt
import numpy as np

r = np.load('eps_rewards.npy')
N = 100
cumsum, moving_aves = [0], []
for i, x in enumerate(r, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves.append(moving_ave)

r_fil = np.array(moving_aves)
fig = plt.figure(figsize=(10,7))
plt.title('Episode Rewards')
plt.plot(r,alpha=.5,label='episode rewards')
plt.plot(r_fil,label='average episode rewards')
plt.legend()
plt.show()

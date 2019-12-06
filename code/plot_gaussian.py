import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def linear_reward_zero(x):
 if(x < 0.45): return 0
 else: return (0.9 - x)

def linear_reward_discont(x):
 if(x < 0.45): return (x - 0.45)
 else: return (0.9 - x)

def linear_reward_peak(x):
 if(x < 0.45): return x
 else: return (0.9 -x)


def quadratic(x):
    return -8 * ((x - 0.45) **2) + 0.4

def Gaussian(x):
    return 0.105*stats.norm.pdf(x, 0.45, 0.1)

# plot piece-wise linear rewards function
x = np.linspace(0, 1, 1000)

reward = [linear_reward_zero, linear_reward_discont, linear_reward_peak, quadratic]

fig = plt.figure()
plt.subplots_adjust(hspace = 0.3)
for i in range(len(reward)):
    y = []
    for j in x:
        y.append(reward[i](j))
    print(i)
    ax = plt.subplot(2,2, i+1)
    plt.axhline(0,linestyle='--')
    plt.plot(x, y)
    ax.set_title("setting "+str(i+1))

fig.savefig('reward_func.png')

# plot quadratic rewards function


# plot Gaussian function




# mu = 0
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 500)
# plt.plot(x, 0.1*stats.norm.pdf(x, 0.45, 0.1))
# plt.plot(x, 0.11*stats.norm.pdf(x, 0.45, 0.1))
# plt.plot(x, 0.105*stats.norm.pdf(x, 0.45, 0.1))
# plt.show()

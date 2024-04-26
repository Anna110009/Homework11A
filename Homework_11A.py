import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
import time

fig, axs = plt.subplots(2, 3, figsize=(11, 6))

num_trials = 500
dice_r = 7
dice_f = 6
means = []
p_values = []


orig_dist = np.random.randint(1, dice_f + 1, size=num_trials * dice_r)
orig_dist_means = np.mean(orig_dist.reshape(-1, dice_r), axis=1)


hist_ax = axs[0, 0]
hist_ax.set_title('the histogram of the means')
hist_ax.set_xlabel('mean')
hist_ax.set_ylabel('frequency')


qq_ax = axs[0, 1]
qq_ax.set_title('QQ Plot')
qq_ax.set_xlabel('theoretical quantiles')
qq_ax.set_ylabel('ordered values')


orig_dist_ax = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
orig_dist_ax.set_title('the original distribution')
orig_dist_ax.set_xlabel('value')
orig_dist_ax.set_ylabel('frequency')
orig_dist_ax.yaxis.set_ticklabels([])


dice_dist_ax =  axs[1, 0]
dice_dist_ax.set_title('shapiro-Wilk test p-values')


p_values_ax = axs[1, 1]
p_values_ax.set_title('shapiro-wilk test p-values')
p_values_ax.set_xlabel('trial')
p_values_ax.set_ylabel('p-value')

shapiro_ax = axs[1, 2]
shapiro_ax.axis('off')

# update plots
def update(frame):
    dice_t = np.random.randint(1, dice_f + 1, size=dice_r)
    means.append(np.mean(dice_t))
    
    
    hist_ax.clear()
    hist_ax.hist(means, bins=6, range=(0, 7), rwidth=0.9)
    hist_ax.set_title('the Histogram of The Means')
    hist_ax.set_xlabel('mean')
    hist_ax.set_ylabel('frequency')
    hist_ax.set_xlim(0, 6)
    hist_ax.set_ylim(0, 20)
 
    
    qq_ax.clear()
    stats.probplot(means, dist="norm", plot=qq_ax)
    qq_ax.set_title('QQ Plot')
    qq_ax.set_xlabel('theoretical Quantiles')
    qq_ax.set_ylabel('sample quantiles')

    
    orig_dist_ax.clear()
    orig_dist_ax.hist(orig_dist_means[:frame], bins=6, range=(0, 7), rwidth=0.9)
    orig_dist_ax.set_title('the original distribution')
    orig_dist_ax.set_xlabel('value')
    orig_dist_ax.set_ylabel('frequency')
    orig_dist_ax.set_xlim(0, 6)
    orig_dist_ax.set_ylim(0, 10)
    
    
    shapiro_statistic, p_value = stats.shapiro(dice_t)
    p_values.append(p_value)
      
    dice_dist_ax.clear()
    dice_dist_ax.set_title('shapiro-wilk test results')
    dice_dist_ax.text(0.5, 0.5, f'P-values: {p_value:.3f}',fontsize=15, ha='center', va='center', transform=dice_dist_ax.transAxes)
    dice_dist_ax.axis('off')
    
    
    p_values_ax.clear()
    p_values_ax.plot(range(len(p_values)), p_values, marker='o', linestyle='-')
    p_values_ax.set_title('shapiro-wilk test p-values')
    p_values_ax.set_xlabel('trial')
    p_values_ax.set_ylabel('p-value')
    p_values_ax.set_ylim(0, 1)
    
    
    
    time.sleep(0.05)

# animation
ani = animation.FuncAnimation(fig, update, frames=num_trials, interval=500)

plt.tight_layout()
plt.show()


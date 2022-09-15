SUPTITLE_SIZE = 16
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 15
DOT_SIZE = 24

# ylim buffer
size = 0.02

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
    
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('axes', labelsize=LABEL_SIZE,titlesize=TITLE_SIZE)
plt.rc('legend',fontsize=LEGEND_SIZE)
plt.rcParams['legend.title_fontsize'] = LEGEND_SIZE
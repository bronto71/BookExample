import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 10000

weight = np.random.randn(N)*5+70
spec_id = np.random.randint(0, 3, N)
bias = [0.9, 1, 1.1]
height = np.array([weight[i]/100 + bias[b] for i, b in enumerate(spec_id)])
spec_name = ['Goblin', 'Human', 'ManBears']
spec = [spec_name[s] for s in spec_id]

colors = ['r', 'b', 'g']
f, axarr = plt.subplots(1, 2, figsize=[7, 3])
ax = axarr[0]
for ii in range(3):
    ax.hist(height[spec_id == ii], 50, color=colors[ii], alpha=0.5)
    ax.set_xlabel('Height')
    ax.set_ylabel('Frequency')
    ax.set_title('Height distribution')

height = height + np.random.randn(N)*0.015
ax.text(1.42, 150, 'Goblins')
ax.text(1.63, 210, 'Humans')
ax.text(1.85, 150, 'ManBears')

ax.set_ylim([0, 260])
ax.set_xlim([1.38, 2.05])

df = pd.DataFrame({'Spcies': spec, 'Weight': weight, 'Height': height})

ax = axarr[1]
ax.plot(df['Height'], df['Weight'], 'o', alpha=0.3, mfc='w', mec='b')
ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.set_title('Heights vs. Weights')

plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight', format='png', dpi=300)

plt.show()
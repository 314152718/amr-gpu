import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('grid.csv')
df['dx'] = 1 / pow(2, df['L'])
df['x'] = df['i'] * df['dx'] + df['dx'] / 2
df['y'] = df['j'] * df['dx'] + df['dx'] / 2
df['z'] = df['k'] * df['dx'] + df['dx'] / 2

plt.plot(list(df['x']), list(df['rho']))
plt.plot(list(df['x']), list(df['rho_grad_x']))
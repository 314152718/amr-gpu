import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read CSV file
df = pd.read_csv("grid.csv")

# Extracting data
i, j, k = df['i'], df['j'], df['k']
rho_grad_x, rho_grad_y, rho_grad_z = df['rho_grad_x'], df['rho_grad_y'], df['rho_grad_z']

print(i)
print(j)
print(k)
print(rho_grad_x)
print(rho_grad_y)
print(rho_grad_z)

# Create a 3D quiver plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(i, j, k, rho_grad_x, rho_grad_y, rho_grad_z, length=0.5, normalize=True, color='b', arrow_length_ratio=0.5)

# Set labels and title
ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel('k')
ax.set_title('3D Quiver Plot')

# Show the plot
plt.show()

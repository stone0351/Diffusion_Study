import numpy as np
import matplotlib.pyplot as plt

tensor = np.array([[[5]]])  # 假设元素值为5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, s=300, c='r')  # 只在原点有一个点
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
ax.set_title('Shape: (1, 1, 1)')
plt.show()
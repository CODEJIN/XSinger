import torch
import matplotlib.pyplot as plt

cosmap_calc = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))

x = torch.linspace(0.0, 1.0, 100)
y = cosmap_calc(x)

print(x)
print(y)

plt.scatter(x, y)
plt.scatter(x, x)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from regression import func, func2, func3

import os

dir = os.getcwd() + "\\assignment\\assignment_01\\"

# Load the data.
data = np.loadtxt(dir + "train_data.csv",
                  delimiter=",",
                  skiprows=1)

x = data[:, 0]
y = data[:, 1]

# Predict the y-values with my function.
x_hat = np.array([i / 1000 for i in range(1000, 101000)])
y_hat = func(x)
y_hat_2 = func(x_hat)
y_hat_3 = func2(x_hat)
y_hat_4 = func3(x_hat)

# Calculate the mean squared error (MSE).
mse = np.mean((y - y_hat)**2)

print("MSE:", mse)


# Plot the data.
fig = plt.figure()
plt.scatter(x, y)
plt.plot(x_hat, y_hat_3, "g-")
plt.plot(x_hat, y_hat_2, "r-")
plt.plot(x_hat, y_hat_4, "b-")


plt.xlabel("x", labelpad=8, fontsize=28)
plt.ylabel("y", labelpad=8, fontsize=28)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


fig.set_size_inches(16, 12)
fig.tight_layout()
fig.savefig(dir + "scatter_with_func.jpg", dpi=300)


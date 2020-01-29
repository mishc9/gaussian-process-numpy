import numpy as np
import matplotlib.pyplot as plt

from utils import kernel, get_train_data

# Test data
n = 50
n_samples = 5
param = 0.25

# Noiseless training data


train_points = [-4, -3, -3.5, -2, -1, 1, 2, 3.5]
x_train, y_train = get_train_data(train_points, np.sin)
x_test = np.linspace(-5, 5, n).reshape(-1, 1)


K_ss = kernel(x_test, x_test, param)

# Get cholesky decomposition (square root) of the
# covariance matrix
B = np.linalg.cholesky(K_ss + 1e-15 * np.eye(n))
# Todo: move prior plot to function
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
# f_prior = np.dot(B, np.random.normal(size=(n, 3)))
# Now let's plot the 3 sampled functions.
# plt.plot(x_test, f_prior)
# plt.axis([-5, 5, -3, 3])
# plt.title('Three samples from the GP prior')
# plt.show()

# Apply the kernel function to our training points
K = kernel(x_train, x_train, param)
L = np.linalg.cholesky(K + 0.00005 * np.eye(len(x_train)))

# Compute the mean at our test points.
K_s = kernel(x_train, x_test, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, y_train)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk ** 2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, n_samples)))

plt.plot(x_train, y_train, 'bs', ms=8)
plt.plot(x_test, f_post)
plt.plot(x_test, np.sin(x_test), c='k')
plt.gca().fill_between(x_test.flat, mu - 2 * stdv, mu + 2 * stdv, color="#dddddd")
plt.plot(x_test, mu, 'r--', lw=2)
plt.axis([-5, 5, -3, 3])
plt.title(f'{n_samples} samples from the GP posterior')
plt.show()

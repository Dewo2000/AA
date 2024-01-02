import numpy as np

def sigmoid(z):
    g= 1/(1+np.exp(-z))

    return g
def sigmoid_deriv(z):
	return sigmoid(z)*(1-sigmoid(z))

def feedforward(thetas, X):
    m = X.shape[0]
    a = [np.array(X)]
    z = []

    for theta in thetas:
        a[-1] = np.hstack([np.ones((m, 1)), a[-1]])
        z.append(np.dot(a[-1], theta.T))
        a.append(sigmoid(z[-1]))

    return a, z

def L2(thetas, y, lambda_):
    m = len(y)
    regularization_term = (lambda_ / (2 * m)) * sum([np.sum(theta[:, 1:]**2) for theta in thetas])
    return regularization_term

def cost_reg(thetas, X, y, lambda_):
    a, _ = feedforward(thetas, X)
    m = len(X)
    J = cost(a[-1], y) + L2(thetas, y, lambda_)
    return J

def cost(a, y):
    m = len(y)
    t1 = np.sum(y * np.log(a))
    t2 = np.sum((1 - y) * np.log(1 - a))
    J = (-1 / m) * np.sum(t1 + t2)
    return J

def backprop(thetas, X, y, lambda_):
    a, z = feedforward(thetas, X)
    m = len(X)
    deltas = [a[-1] - y]

    for i in range(len(thetas)-1, 0, -1):
        delta = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_deriv(z[i-1])
        deltas.insert(0, delta)

    grads = [(1 / m) * np.dot(deltas[i].T, a[i]) for i in range(len(deltas))]
    grads = [grad + (lambda_ / m) * np.hstack([np.zeros((grad.shape[0], 1)), theta[:, 1:]]) for grad, theta in zip(grads, thetas)]

    J = cost_reg(thetas, X, y, lambda_)

    return J, grads

def gradientDescentTraining(thetas, X, y, lambda_, alpha, num_iters):
    e = 0.12
    for i in range(len(thetas)):
        thetas[i] = np.random.uniform(-e, e, size=thetas[i].shape)

    for _ in range(num_iters):
        J, grads = backprop(thetas, X, y, lambda_)
        thetas = [theta - alpha * grad for theta, grad in zip(thetas, grads)]

    return thetas

def predict(thetas, X):
    m = X.shape[0]
    a, _ = feedforward(thetas, X)
    p = np.argmax(a[-1], axis=1)

    return p
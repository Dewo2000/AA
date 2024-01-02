import numpy as np

def sigmoid(z):
    g= 1/(1+np.exp(-z))

    return g
def sigmoid_deriv(z):
	return sigmoid(z)*(1-sigmoid(z))

def feedforward(thetas, X):
    m = 3
    a = []
    z = []

    a1= np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    a.append(a1)

    for i in thetas:
        z2 = np.dot(a1, thetas[i].T)
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
        z.append(z2)
        a.append(a2)
        a1=a2
        
    z3 = np.dot(a2, thetas[m-1].T)
    a3 = sigmoid(z3)
    z.append(z3)
    a.append(a3)
    return a, z

def L2(thetas, y, lambda_):
    m = len(y)
    regularization_term = (lambda_/(2*m))*np.sum(np.sum(np.sum(thetas**2)))
    return regularization_term

def cost_reg(neuronas_por_capas,thetas, X, y, lambda_):
    J = cost(thetas,X,y)
    m = len(X)
    thetas = np.concatenate([theta[:,1:].flatten() for theta in thetas])
    Jr = J+L2(thetas,y,lambda_)
    return Jr

def cost(thetas,X, y):
    a, z = feedforward(thetas, X)
    m = len(y)
    t1 = np.sum(y * np.log(a))
    t2 = np.sum((1 - y) * np.log(1 - a))
    J = (-1 / m) * np.sum(t1 + t2)
    return J

def backprop(neuronas_por_capas,thetas, X, y, lambda_):
    a, z = feedforward(thetas, X)
    m = len(X)
    deltas = [a[-1] - y]

    for i in range(len(neuronas_por_capas)-1, 0, -1):
        delta = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_deriv(z[i-1])
        deltas.insert(0, delta)

    grads = [(1 / m) * np.dot(deltas[i].T, a[i]) for i in range(len(deltas))]
    grads = [grad + (lambda_ / m) * np.hstack([np.zeros((grad.shape[0], 1)), theta[:, 1:]]) for grad, theta in zip(grads, thetas)]

    J = cost_reg(thetas, X, y, lambda_)

    return J, grads

def gradientDescentTraining(neuronas_por_capas, X, y, lambda_, alpha, num_iters):
    e = 0.12
    thetas = [np.random.uniform(-e, e, size=(neuronas_por_capas[i + 1], neuronas_por_capas[i] + 1))
              for i in range(len(neuronas_por_capas)-1)]#

    for it in range(num_iters):
        J, grads = backprop(neuronas_por_capas,thetas, X, y, lambda_)
        thetas = thetas - (alpha*grads)

    return thetas

def predict(thetas, X):
    m = X.shape[0]
    a, z = feedforward(thetas, X)
    p = np.argmax(a[-1], axis=1)

    return p
import numpy as np
from scipy.special import softmax

num_of_classes = 10
hidden_layer_size = 500


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    if x <= 0:
        return 0
    return 1


def ng_log_likelihood_loss(y, h2):
    return -(y * np.log(h2) + (1 - y) + np.log(1 - h2))


def load_data():
    train_x = np.loadtxt('train_x_short')
    train_y = np.loadtxt('train_y_short')
    return train_x, train_y


def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = relu(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = ng_log_likelihood_loss(y, h2)
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    dz2 = (h2 - y)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 (h2 - y)) * relu_derivative(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def main():
    y_hat = []
    train_x, train_y = load_data()
    d = train_x.shape[1]
    W1 = np.zeros((hidden_layer_size, d))
    b1 = np.ones((hidden_layer_size, 1))

    W2 = np.zeros((num_of_classes, hidden_layer_size))
    b2 = np.ones((num_of_classes, 1))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    for e in range(0, 2):
        for x, y in zip(train_x, train_y):
            fprop_cache = fprop(x, y, params)
            params = bprop(fprop_cache)
            if e == 1:
                y_hat.append(fprop_cache['h2'])
    errors = 0
    for i in range(len(train_y)):
        if train_y[i] != y_hat[i]:
            errors += 1
    print('precision: {0}'.format(1 - (float(errors) / train_x.shape[0])))


if __name__ == "__main__":
    main()

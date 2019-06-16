import numpy as np

num_of_classes = 10
hidden_layer_size = 100
lr = 0.001


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def load_data():
    train_x = np.loadtxt('train_x', dtype=float)
    train_y = np.loadtxt('train_y')
    return train_x, train_y


def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = relu(z1)

    z2 = np.dot(W2, h1) + b2
    y_hat = stable_softmax(z2)

    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, y_hat = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'y_hat')]

    dz2 = y_hat
    dz2[int(y)] -= 1
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2

    dz1 = np.dot(y_hat.T, fprop_cache['W2']).T * relu_derivative(z1)
    dW1 = np.dot(dz1, x.T)
    db1 = dz1

    return {'db1': db1, 'dW1': dW1, 'db2': db2, 'dW2': dW2}


def update(params, bprop_cache):
    w1, b1, w2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    dw1, db1, dw2, db2 = [bprop_cache[key] for key in ('dW1', 'db1', 'dW2', 'db2')]

    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    params = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    return params


def train(train_x, train_y, params):
    loss = 0
    for x, y in zip(train_x, train_y):
        # make a copy of the array
        x = np.ndarray.astype(x, dtype=float)
        # normalize
        x /= 255
        # reshape to vector
        x = x.reshape(-1, 1)

        fprop_cache = fprop(x, y, params)

        # compute loss
        loss -= np.log(fprop_cache['y_hat'][int(y)])

        bprop_cache = bprop(fprop_cache)
        params = update(fprop_cache, bprop_cache)
    return loss, params


def validation(validation_x, validation_y, params):
    validation_loss = 0
    validation_sucess = 0
    for x, y in zip(validation_x, validation_y):
        # make a copy of the array
        x = np.ndarray.astype(x, dtype=float)
        # normalize
        x /= 255
        # reshape to vector
        x = x.reshape(-1, 1)

        fprop_cache = fprop(x, y, params)

        # compute success
        if y == np.argmax(fprop_cache['y_hat']):
            validation_sucess += 1

        # compute loss
        validation_loss -= np.log(fprop_cache['y_hat'][int(y)])
    return validation_sucess, validation_loss


def shuffle_data(x, y):
    # every day I'm shuffling
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def split_to_validation(x, y, split_ratio=0.8):
    x, y = shuffle_data(x, y)
    split_index = int(x.shape[0] * split_ratio)
    train_x, train_y, validation_x, validation_y = x[:split_index, :], y[:split_index], \
                                                   x[split_index:, :], y[split_index:]
    return train_x, train_y, validation_x, validation_y


def main():
    x, y = load_data()
    d = x.shape[1]
    input_size = x.shape[0]

    W1 = np.random.uniform(low=-0.08, high=0.08, size=(hidden_layer_size, d))
    b1 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_layer_size, 1))

    W2 = np.random.uniform(low=-0.15, high=0.15, size=(num_of_classes, hidden_layer_size))
    b2 = np.random.uniform(low=-0.5, high=0.5, size=(num_of_classes, 1))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    split_ratio = 0.8
    train_x, train_y, validation_x, validation_y = split_to_validation(x, y, split_ratio)

    for e in range(0, 30):
        train_x, train_y = shuffle_data(train_x, train_y)
        loss, params = train(train_x, train_y, params)
        validation_success, validation_loss = validation(validation_x, validation_y, params)

        avg_loss = loss / (input_size * split_ratio)
        avg_validation_loss = validation_loss / (input_size * (1 - split_ratio))
        validation_acc = validation_success / (input_size * (1 - split_ratio))
        print('# {0} acc {1} loss {2} validation_loss {3}'.format(e, round(validation_acc, 3), avg_loss[0],

                                                                  avg_validation_loss[0]))
    predictions = []
    test_x = np.loadtxt('test_x')
    for example in test_x:
        # make a copy of the array
        example = np.ndarray.astype(example, dtype=float)
        # normalize
        example /= 255
        # reshape to vector
        example = example.reshape(-1, 1)
        fprop_cache = fprop(example, 0, params)
        prediction = int(np.argmax(fprop_cache['y_hat']))
        predictions.append(prediction)

    np.savetxt(fname='test_y', X=np.asarray(predictions), fmt='%d')


if __name__ == "__main__":
    main()

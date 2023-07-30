from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# constants
ITERATIONS = 10_000
DISPLAY_REG = 50
IMG_SIZE = 784
DATASET_PARTITION = 1_000
TEST_PREDICTIONS = 5

# read data in
data = pd.read_csv("mnist_sample.csv")
# prepare data, collect its dimensions & shuffle
data = np.array(data)
rows, cols = data.shape
np.random.shuffle(data)

# prepare cross-validation data
# note: we transpose all matrices that we feed the model
#       this allows us to separate the labels and images easier by simply
#       extracting the first row
data_crossv = data[0:1000].T
y_crossv = data_crossv[0]
x_crossv = data_crossv[1:cols]
x_crossv = x_crossv / 255.

data_train = data[1000:rows].T
y_train = data_train[0]
x_train = data_train[1:cols]
x_train = x_train / 255.
_, m_train = x_train.shape

# starting parameters for the neural network
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# using ELU rather ReLU
def ELU(Z, a=1.0):
    return np.where(Z <= 0, a * (np.exp(Z) - 1), Z)

def ELU_deriv(Z, a=1.0):
    return np.where(Z <= 0, a * np.exp(Z), 1)

def softmax(Z):
    prob = np.exp(Z) / sum(np.exp(Z))
    return prob

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ELU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / rows * dZ2.dot(A1.T)
    db2 = 1 / rows * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ELU_deriv(Z1)
    dW1 = 1 / rows * dZ1.dot(X.T)
    db1 = 1 / rows * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)

    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % DISPLAY_REG == 0:
            print("\nIteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, ITERATIONS)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

for x in range(TEST_PREDICTIONS):
    test_prediction(x, W1, b1, W2, b2)

dev_predictions = make_predictions(x_crossv, W1, b1, W2, b2)
print("Deployed accuracy: ", round(get_accuracy(dev_predictions, y_crossv) * 100, 3), "%\n")
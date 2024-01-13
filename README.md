# MNIST
![PyPI - Version](https://img.shields.io/pypi/v/numpy)
![Static Badge](https://img.shields.io/badge/dataset-mnist-purple)
![Static Badge](https://img.shields.io/badge/neural%20network-vanilla-yellow)

This project is a 2 layer neural network that learns to recognise handwritten numbers by being fed the famous [MNIST dataset](https://huggingface.co/datasets/mnist) as a .csv. This project uses no tensorflow, pytorch etc., solely math.

---------------------------------------
* [Settings](#settings)
* [Forward Propagation](#forward-propagation)
* [Backward Propagation](#backward-propagation)
* [Update Parameters](#update-parameters)

---------------------------------------

## Settings
The script includes a set of tweakable constants:
* `ITERATIONS`: How many iterations the neural network will loop.
* `DISPLAY_REG`: How often the script will display the neural network's current accuracy.
* `IMG_SIZE`: The number of pixels in the input image.
* `DATASET_PARTITION`: Where the data should be parted for crossvalidation.
* `TEST_PREDICTIONS`: How many times the neural network will be tried for predictions upon completing training.
* `DATASET_FILE`: Input path for the data.

## Forward Propagation
During forward propagation, the neural network takes images and learns to create predictions out of them:
* A<sup>0</sup>: This is the input layer (layer 0) of the neural network. It simply receives the `IMAGE_SIZE` number of pixels into each node.

* Z<sup>1</sup>: Unactivated first layer. Z<sup>1</sup> is obtained by applying a weight obtained from the connections between nodes in the prior layer (W<sup>1</sup>) and a bias (b<sup>1</sup>) to the input layer (A<sup>0</sup>). Or, Z<sup>1</sup> = W<sup>1</sup> * A<sup>0</sup> + b<sup>1</sup>.
* A<sup>1</sup>: First layer. A<sup>1</sup> is obtained by putting Z<sup>1</sup> through an activation function. The activation function I use is Exponential Linear Unit, or ELU.

* Z<sup>2</sup>: Unactivated second layer. Z<sup>2</sup> is obtained by applying a weight obtained from the connections between nodes in the prior layer (W<sup>2</sup>) and a bias (b<sup>2</sup>) to the prior layer (A<sup>1</sup>). Or, Z<sup>2</sup> = W<sup>2</sup> * A<sup>1</sup> + b<sup>2</sup>.
* A<sup>2</sup>: Second and final layer. A<sup>2</sup> is obtained from passing Z<sup>2</sup> through an activation function. This time we're using softmax, which will assign a probability to each node in this output layer.

## Backward Propagation
Backward propagation is a method of improving the algorithm as it learns. This is done by taking the prediction, measuring how much it deviated from the image's label and working backwards.
* dZ<sup>2</sup>: A measure of the error in the second layer. It's obtained by taking the predictions and subtracting the labels from them. For that, we one-hot encode the label as Y. dZ<sup>2</sup> = A<sup>2</sup> - Y
* dW<sup>2</sup>: The derivative of the loss function with respect to the weights in layer 2. dW<sup>2</sup> = 1/m * dZ<sup>2</sup> * A<sup>1</sup>.T. (Where .T is transposition of a matrix or vector)
* db<sup>2</sup>: This is the average of the absolute error. db<sup>2</sup> = 1/m * Σ dZ<sup>2</sup>.

* dZ<sup>1</sup>: A measure of the error in the first layer. This formula essentially performs forward propagation in reverse. dZ<sup>1</sup> = W<sup>2</sup>.T * dZ<sup>1</sup> * g'() where g'() is the derivative of the activation function.
* dW<sup>1</sup>: The derivative of the loss function with respect to the weights in layer 1. dW<sup>1</sup> = 1/m * dZ<sup>1</sup> * X.T.
* db<sup>1</sup>: db<sup>1</sup> = 1/m * Σ dZ<sup>1</sup>.

## Update Parameters
After successful forward & backward propagation, the algorithm updates a hyperparameter α in this fashion:
* W<sup>1</sup> := W<sup>1</sup> - αdW<sup>1</sup>
* b<sup>1</sup> := b<sup>1</sup> - αdb<sup>1</sup>
* W<sup>2</sup> := W<sup>2</sup> - αdW<sup>2</sup>
* b<sup>2</sup> := b<sup>2</sup> - αdb<sup>2</sup>

α, being a hyperparameter, isn't set by gradient descent, but by the end-user. α can be interpreted as the learning rate.

After that, the algorithm loops back to forward propagation.

![Example](https://github.com/raneamri/mnist/blob/master/img/example.png)
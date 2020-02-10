from mnist import MNIST
import random

mndata = MNIST('data')

images, labels = mndata.load_training()
# or
images, labels = mndata.load_testing()

# images = 784 or 28x28 in a single vector
print(labels[0])
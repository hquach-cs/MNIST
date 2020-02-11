from mnist import MNIST
import numpy as np 


# Load MNIST Data from http://yann.lecun.com/exdb/mnist/
mndata = MNIST('data')
# images[i] = 784 or 28x28 in a single vector format
# Labels[i] = number
images, labels = mndata.load_training()
#images, labels = mndata.load_testing()

# Hyperparameters
learning_rate = 0.25
epoch = 1000

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Init
inputs = np.array(images)
actual_output = []
for i in labels:
    temp = [0,0,0,0,0,0,0,0,0,0]
    temp[i] = 1 
    actual_output.append(temp)
actual_output = np.array(actual_output)
inputLayerNeurons,outputLayerNeurons = 784,10
# hiddenLayerNeurons = 0 If no hiddenlayer
hiddenLayerNeurons = 784
hiddenLayerCnt = len(hiddenLayerNeurons) if type(hiddenLayerNeurons) == list else 1
weights,bias = [],[]
if(hiddenLayerNeurons == 0):
    hiddenLayerCnt = 0
    weights.append(np.random.uniform(size=(inputLayerNeurons,outputLayerNeurons)))
    bias.append(np.random.uniform(size=(1,outputLayerNeurons)))
elif(hiddenLayerCnt == 1):
    weights.append(np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons)))
    weights.append(np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons)))
    bias.append(np.random.uniform(size=(1,hiddenLayerNeurons)))
    bias.append(np.random.uniform(size=(1,outputLayerNeurons)))
else:
    weights.append(np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons[0])))
    bias.append(np.random.uniform(size=(1,hiddenLayerNeurons[0])))
    for i in range(hiddenLayerCnt-1):
        weights.append(np.random.uniform(size=(hiddenLayerNeurons[i],hiddenLayerNeurons[i+1])))
        bias.append(np.random.uniform(size=(1,hiddenLayerNeurons[i+1])))
    weights.append(np.random.uniform(size=(hiddenLayerNeurons[hiddenLayerCnt-1],outputLayerNeurons)))
    bias.append(np.random.uniform(size=(1,outputLayerNeurons)))
for _ in range(epoch):
    # Feedforward
    hidden_layer_activation = [np.dot(inputs,weights[0]) + bias[0]]
    hidden_layer = [sigmoid(hidden_layer_activation[0])]
    for i in range(hiddenLayerCnt):
        hidden_layer_activation.append(np.dot(hidden_layer[i],weights[i+1]) + bias[i+1])
        hidden_layer.append(sigmoid(hidden_layer_activation[i+1]))
    predicted_output = hidden_layer[hiddenLayerCnt]

    # Backpropagation
    error = (actual_output - predicted_output)**2
    d_weights,d_bias = [],[]
    d_weights.append(2 * (actual_output - predicted_output) * sigmoid_derivative(predicted_output))
    for i in range(hiddenLayerCnt):
        d_weights.append(np.dot(d_weights[0],weights[hiddenLayerCnt-i].T) * sigmoid_derivative(hidden_layer[hiddenLayerCnt-1-i]))
    d_weights.reverse()
    weights[0] += np.dot(inputs.T,d_weights[0]) * learning_rate
    bias[0] += np.sum(d_weights[0],axis=0,keepdims=True) * learning_rate
    for i in range(1,len(weights)):
        weights[i] += np.dot(hidden_layer[i-1].T,d_weights[i]) * learning_rate
        bias[i] += np.sum(d_weights[i],axis=0,keepdims=True) * learning_rate

print(actual_output[:10])
print(predicted_output[:10])
    
import numpy as np
import pandas as pd

epochs = 10000
size_of_batches = 30
learningRate = 0.0002

df = pd.read_csv('IRIS.csv')
for i in range(20):
    df = df.sample(frac=1)

df = df.to_numpy()

labels_csv = df[:, 4]

classes = np.unique(labels_csv)

labels_formatted = np.empty(labels_csv.shape[0])

for i in range(len(labels_csv)):
    if labels_csv[i] == classes[0]:
        labels_formatted[i] = 0
    if labels_csv[i] == classes[1]:
        labels_formatted[i] = 1
    if labels_csv[i] == classes[2]:
        labels_formatted[i] = 2

labels_formatted = np.reshape(labels_formatted, newshape=(1, len(labels_csv)))

input_values = df[:, 0:4]
input_values = input_values.T

layer1_weights = np.random.random(size=(4, 4))
layer2_weights = np.random.random(size=(2, 4))
layer3_weights = np.random.random(size=(1, 2))

layer1_biases = np.random.random(size=(4, 1))
layer2_biases = np.random.random(size=(2, 1))
layer3_biases = np.random.random(size=(1, 1))


def train_network(l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, inputs, labels):
    for _ in range(epochs):
        loop_cost = 0
        for i in range(len(inputs)):
            x = inputs[i]
            y = labels[i]

            l1_o, l2_o, prediction = feed_forward(x, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)

            loss = loss_function(prediction, y)
            loss_difference = prediction - y

            l3_w = np.subtract(l3_w, learningRate * np.matmul(loss_difference, l2_o.T))
            l3_b = np.subtract(l3_b, learningRate * loss)

            l2_w = np.subtract(l2_w, learningRate * np.matmul(l3_w.T, np.matmul(loss_difference, l1_o.T)))
            l2_b = np.subtract(l2_b, learningRate * np.matmul(loss_difference, l2_o.T).T)

            l1_w = np.subtract(l1_w,
                               learningRate * np.matmul(l2_w.T, np.matmul(np.matmul(x, loss_difference.T), l3_w).T))
            l1_b = np.subtract(l1_b, learningRate * np.matmul(l2_w.T, np.matmul(loss_difference, l2_o.T).T))

            loop_cost += loss

        print(loop_cost)
    return l1_w, l1_b, l2_w, l2_b, l3_w, l3_b


def feed_forward(x, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b):
    layer1_output = np.matmul(l1_w, x) + l1_b
    layer2_output = np.matmul(l2_w, layer1_output) + l2_b
    layer3_output = np.matmul(l3_w, layer2_output) + l3_b

    return layer1_output, layer2_output, layer3_output


def loss_function(prediction, label):
    return np.average(np.square(prediction - label).astype(np.float))


input_values_batches = np.split(input_values, size_of_batches, axis=1)
input_values_train = np.array(input_values_batches[0:2])
input_values_test = np.array(input_values_batches[3:5])

labels_formatted_batches = np.split(labels_formatted, size_of_batches, axis=1)
labels_formatted_train = np.array(labels_formatted_batches[0:2])
labels_formatted_test = np.array(labels_formatted_batches[3:5])

l1_w, l1_b, l2_w, l2_b, l3_w, l3_b = train_network(layer1_weights, layer1_biases, layer2_weights, layer2_biases,
                                                   layer3_weights, layer3_biases,
                                                   input_values_train, labels_formatted_train)

# Testing data tested below
print("Testing")
output = np.around(
    np.array(feed_forward(input_values_test, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)[2]).astype(np.double))

print(loss_function(labels_formatted_test, output))

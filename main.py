import numpy as np
import pandas as pd

epochs = 20000
size_of_batches = 50
learningRate = 0.000000003

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
    for e in range(epochs):
        loop_cost = 0
        for i in range(len(inputs)):
            x = inputs[i]
            y = labels[i]

            l1_o, l2_o, prediction = feed_forward(x, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)

            loss = loss_function(prediction, y)
            loss_difference = prediction - y

            l3_w_delta = loss_difference.dot(l2_o.T)
            l2_w_delta = l2_o.dot(np.sum(l3_w.T.dot(l3_w_delta)))
            l1_w_delta = l1_o.dot(np.sum(l2_w.T.dot(l2_w_delta)))

            l3_w = np.subtract(l3_w, learningRate * l3_w_delta)
            l3_b = np.subtract(l3_b, learningRate * np.sum(l3_w_delta))

            l2_w = np.subtract(l2_w, learningRate * l2_w_delta.dot(l1_o.T))
            l2_b = np.subtract(l2_b, learningRate * np.sum(l2_w_delta))

            l1_w = np.subtract(l1_w, learningRate * l1_w_delta.dot(x.T))
            l1_b = np.subtract(l1_b, learningRate * np.sum(l1_w_delta))

            loop_cost += loss

        if e % 500 == 0:
            print(str(e) + ":", loop_cost)

    return l1_w, l1_b, l2_w, l2_b, l3_w, l3_b


def feed_forward(x, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b):
    layer1_output = np.matmul(l1_w, x) + l1_b
    layer2_output = np.matmul(l2_w, layer1_output) + l2_b
    layer3_output = np.matmul(l3_w, layer2_output) + l3_b

    return layer1_output, layer2_output, layer3_output


def loss_function(prediction, label):
    return np.average(np.square(prediction - label).astype(np.float))


input_values = input_values / (input_values.max() - input_values.min())

size_of_batches = 15

training_batch_size = 6

input_values_batches = np.split(input_values, size_of_batches, axis=1)
input_values_train = np.array(input_values_batches[0:training_batch_size])
input_values_test = np.array(input_values_batches[training_batch_size:size_of_batches])

labels_formatted_batches = np.split(labels_formatted, size_of_batches, axis=1)
labels_formatted_train = np.array(labels_formatted_batches[0:training_batch_size])
labels_formatted_test = np.array(labels_formatted_batches[training_batch_size:size_of_batches])

learningRate = 0.000007
epochs = 50000

l1_w, l1_b, l2_w, l2_b, l3_w, l3_b = train_network(layer1_weights, layer1_biases, layer2_weights, layer2_biases,
                                                   layer3_weights, layer3_biases,
                                                   input_values_train, labels_formatted_train)

# Testing data tested below
print("Testing")

output = np.array(feed_forward(input_values_test, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)[2]).astype(np.double)
print(loss_function(labels_formatted_test, output))

accuracy = 0


for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        for k in range(output.shape[2]):
            if (output[i][j][k] - labels_formatted_test[i][j][k] < 0.2):
                accuracy += 1

print("Accuracy:", accuracy * 100 / output.size)

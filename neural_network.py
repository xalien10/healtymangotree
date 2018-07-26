import numpy as np


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


# Run the network

epochs = 3

ANN = NeuralNetwork(no_of_in_nodes=4, no_of_out_nodes=1, no_of_hidden_nodes=100, learning_rate=0.1)

data = np.loadtxt("xor.csv", delimiter=",")

train_data = data[:10, :4]
train_target = data[:10, 4:]
test_data = data[11:, :4]
test_target = data[11:, 4:]
for epoch in range(epochs):
    ANN.train(train_data, train_target)
    corrects, wrongs = ANN.evaluate(train_data, train_target)
    print("accruracy train: ", corrects / (corrects + wrongs))
    corrects, wrongs = ANN.evaluate(test_data, test_target)
    print("accruracy: test", corrects / (corrects + wrongs))

# Testing the network
for i in range(len(test_data)):
    res = ANN.run(test_data[i])
    print(test_target[i], np.argmax(res), np.max(res))

corrects, wrongs = ANN.evaluate(train_data, train_target)
print("accruracy train: ", corrects / (corrects + wrongs))
corrects, wrongs = ANN.evaluate(test_data, test_target)
print("accruracy: test", corrects / (corrects + wrongs))
cm = ANN.confusion_matrix(train_data, train_target)
print(cm)
for i in range(1):
    print("output: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))

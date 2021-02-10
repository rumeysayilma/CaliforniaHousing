import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt

f = open("SonHousing.csv", "r")
y_d = []
x = []

for row in f:
    rowsArray = row.split(',')
    y_d.append(np.array(rowsArray[2]).astype(np.float))
    x.append(np.array([rowsArray[0],rowsArray[1],rowsArray[3],rowsArray[4],rowsArray[5]]).astype(np.float).T)


def data_vectorizer(data_to_vector, n):
    final = [a.reshape((n, 1)) for a in data_to_vector]
    return final

def datayi_test_ve_egitimi_ayir(x, y):
 
    #datanın yüzde 30 u test olarak alınır
    x_egitim = x[0:12800]
    x_test = x[12800:18286]
    yd_egitim = y[0:12800]
    yd_test = y[12800:18286]
    return x_egitim, x_test, yd_egitim, yd_test

x_egitim, x_test, yd_egitim, yd_test = datayi_test_ve_egitimi_ayir(
    x, y_d)
x_egitim = data_vectorizer(x_egitim, 5)
x_test = data_vectorizer(x_test, 5)


Network = NeuralNetwork.NeuralNetwork([5, 4, 3, 1])
epoch, loss, test_loss, test_accuracies, train_accuracies = Network.train(x_train=x_egitim, y_train=yd_egitim, x_test=x_test,
                                                                              y_test=yd_test, epochs=30, learning_rate=0.6, alfa=0.6, tqdm_=True, stop_error=0.00001)

print(test_loss)


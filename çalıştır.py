import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt


""" Reading saved Data  """
X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')


allTestLosses = []
allTrainLosses = []
allTestAccuracies = []
allTrainAccuracies = []

for i in range(11):
    Network = NeuralNetwork.NeuralNetwork([50, 12-i, 4])
    epoch, loss, test_loss, test_accuracies, train_accuracies = Network.train(x_train=X_train, y_train=y_desired, x_test=X_test,
                                                                              y_test=y_test_desired, epochs=100, learning_rate=0.6, alfa=0.6, tqdm_=True, stop_error=0.00001)

    allTrainLosses.append(test_loss)
    allTestLosses.append(loss)
    allTestAccuracies.append(test_accuracies)
    allTrainAccuracies.append(train_accuracies)


# Gerekli Görsellemeler yapılarak sonuçlar yorumlanır
for i in range(len(allTrainLosses)):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Sol Test - Sağ Eğitim ,' + str(11-i) +
                 ' adet ara nöron için Karesel ortalama hata')
    ax1.plot(allTrainLosses[i])

    ax2.plot(allTestLosses[i])
    plt.show()

    plt.plot(allTestAccuracies[i])
    plt.title(str(11-i) + ' adet ara nöron için Test Datası Doğruluğu')
    plt.xlabel("Data ")
    plt.ylabel("Test Datası İçin Doğruluk, ortalama = %" +
               str(np.mean(allTestAccuracies[i])))
    plt.show()

    plt.plot(range(len(allTrainAccuracies[i])), allTrainAccuracies[i], "ro")
    plt.title(str(11-i) + ' kadar ara nöron için Eğitim Datası Doğruluğu')
    plt.xlabel("Data ")
    plt.ylabel("Eğitim Datası için Doğruluk")
    plt.show()

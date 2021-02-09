import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
# from sklearn.metrics import accuracy_score
from tqdm import tqdm

# relu aktivasyon fonksiyonu.

'''def activation(z, derivative=False):
    if derivative:
        squarer1 = lambda z: 1 if z>0  else 0
        squares1 = np.array([squarer1(zi) for zi in z])
        return squares1      
    else:
        squarer2 = lambda z: z if z>0  else 0
        squares2 = np.array([squarer2(zi) for zi in z])  
        return squares2 '''

# Obtain array of square of each element in x


#sigmoid
def activation(z, derivative=False):
    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))         


#mutlak hata hesaplanıyor
def cost_function(y_true, y_pred):
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred.T) ** 2)
    return cost

'''def cost_function(y_true, y_pred):
    cost = mean_absolute_error(y_true,y_pred)
    return cost'''

class NeuralNetwork(object):
    '''
    size örnek olarak [4,3,2] ise bunun anlamı:
    4 nöron giriş
    3 nöron gizli katman
    2 nöron çıkış
    '''

    def __init__(self, size):

        self.size = size
        '''
        Belirtilen size'a uygun olarak
        ilk ağırlıklar ve biase'ler rastgele
        elemanlardan oluşturuluyor.
        '''
        self.weights = [np.random.randn(
            self.size[i], self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

    def forward(self, input):
        '''
        Bütün inputlar, bütün w'lar ile sırayla
        çarpılarak ilerleniyor.
        y değerleri ve v değerleri
        gradyen hesabı için kaydediliyor
        '''
        a = input
        v_values = []
        y_values = []
        y_values.append(a)
        # Burada w,b o katmana ait w,b'yi temsil ediyor
        for z, (w, b) in enumerate(zip(self.weights, self.biases)):
            v = np.dot(w, y_values[z]) + b
            a = activation(v)
            v_values.append(v)
            y_values.append(a)
        # return edilen a değeri y_output değerleri olacaktır.
        return (a, v_values, y_values)

    def compute_gradients(self, v_values, y_d, y_output_values):

        # önce hata hesaplanıyor
        e = (np.reshape(y_d, (self.size[-1], 1)) - y_output_values[0].T)

        # çıkış gradyanı bulundu ve tüm gradyanları içerecek kümeye eklendi
        # toplam gradyan sayısı size-1 olacaktır.
        gradients = [0] * (len(self.size)-1)
        gradients[-1] = e * activation(v_values[-1].T, derivative=True).T

        # son gradyenden geriye doğru gidilerek diğer gradyenler bulunuyor.
        for l in range(len(gradients) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), gradients[l + 1]
                           ) * activation(v_values[l], derivative=True)
            gradients[l] = delta
        return(gradients)

    def backpropagate(self, gradients, v_values, y_values):

        dW = []
        db = []
        gradients = [0] + gradients
        # Bias ve ağırlıkta oluşacak değişimler bulunuyor ve dw,db'de tutuluyor.
        for l in range(1, len(self.size)):
            dW_l = np.dot(gradients[l], y_values[l-1].transpose())
            db_l = gradients[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def train(self, x_train, y_train, x_test, y_test, epochs, learning_rate, alfa, tqdm_=True, stop_error=0.01):
        # momentum terimi için ağırlıklar tutuluyor.
        allWeights = []
        # boş hata ve accuracy listeleri
        all_train_loss = []
        all_train_accuracies = []
        # tqdm isteğe bağlı..
        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            train_losses = []
            train_accuracies = []

            for i, (a, y_d) in enumerate(zip(x_train, y_train)):
                # forward -> gradients -> backward -> test -> update weight,biases
                y_train_pred, v_values, y_values = self.forward(a)
                gradients = self.compute_gradients(
                    v_values, y_d, y_train_pred)
                dW, db = self.backpropagate(gradients, v_values, y_values)

                # loss ve accuracy hesaplanıyor.
                train_loss = cost_function(y_d, y_train_pred)
                train_losses.append(train_loss)
                if self.size[-1] == 1:
                    train_accuracy = (np.sum(np.equal((y_d - y_train_pred),
                                                      np.zeros((1, self.size[-1])))))*100 / (self.size[-1])
                else:
                    train_accuracy = (np.sum(np.equal((y_d.T - y_train_pred.T),
                                                      np.zeros((1, self.size[-1])))))*100 / (self.size[-1])
                train_accuracies.append(train_accuracy)

                allWeights.append([])
                # weight update
                for k, (dw_each, db_each) in enumerate(zip(dW, db)):

                    allWeights[-1].append(dw_each)
                    if i > 1:
                        self.weights[k] = self.weights[k] + learning_rate * \
                            dw_each - alfa * \
                            (allWeights[-1][k] - allWeights[-2][k])
                    else:
                        self.weights[k] = self.weights[k] + \
                            learning_rate * dw_each

                    self.biases[k] = self.biases[k] - learning_rate * db_each
                # print(allWeights)
            # her epochta loss ve accuracy ortalaması  alınıp listeye ekleniyor.
            if(np.mean(train_losses) <= stop_error):
                print('we dont need more education: '+str(e))

            all_train_loss.append(np.mean(train_losses))
            all_train_accuracies.append(np.mean(train_accuracy))

        # Test kümesi -> eğitim sonrası test ediliyor.
        test_losses = []
        test_accuracies = []
        for i, (x_test, y_test) in enumerate(zip(x_test, y_test)):
            y_test_pred, accuracy_pred = self.predict(x_test)
            # tahmini y değerinden gerçek y değeri çıkarılıp farkları incelendi
            # sıfır olanlar için doğru tahmin ,farklı olanlar yanlıştır.
            accuracy = (np.sum(np.equal((y_test.T - accuracy_pred.T),
                                        np.zeros((1, self.size[-1])))))*100 / self.size[-1]
            test_accuracies.append(accuracy)
            test_loss = cost_function(y_test, y_test_pred)
            test_losses.append(np.mean(test_loss))

        return epochs, all_train_loss, test_losses, test_accuracies, all_train_accuracies

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        # [0.3,0.4,.0.8] -> [0,0,1]
        # Değeri en yüksek nöron aktif.
        # Diğerleri ise sıfır .
        accuracy_pred = np.where(a == np.max(a), 1, 0)
        return a, accuracy_pred

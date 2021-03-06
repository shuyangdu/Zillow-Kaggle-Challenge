from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import regularizers
from matplotlib import pyplot as plt


class DNN(object):
    """
    Dense Neural Networks.
    """

    def __init__(self, dim_hidden_lst=(), learning_rate=0.01, decay=0.0, reg=0.01,
                 batch_size=128, epochs=5, verbose=0):
        """
        Constructor
        :param dim_hidden_lst: number of neurons in each hidden layer
        :param learning_rate: learning rate for gradient descent
        :param decay: decay rate for learning rate
        :param reg: regularizer scale
        :param batch_size: batch size for mini-batch training
        :param epochs: epochs of training
        :param verbose: verbose level
        """
        self.dim_inputs = None  # inferred in fit method
        self.dim_hidden_lst = list(dim_hidden_lst)
        self.learning_rate = learning_rate
        self.decay = decay
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        self.train_history = None

    def _init_structure(self):
        """
        Initialize model structure.
        :return: None
        """
        self.model = Sequential()

        # combine input dim, hidden layer dim and output dim together
        dim_lst = [self.dim_inputs] + self.dim_hidden_lst + [1]
        for i in range(len(dim_lst) - 1):
            dim_inputs = dim_lst[i]
            units = dim_lst[i+1]

            layer = Dense(units=units, activation='relu', input_dim=dim_inputs,
                          kernel_regularizer=regularizers.l2(self.reg))
            self.model.add(layer)

    def _init_learning(self):
        """
        Initialize learning process.
        :return: None
        """
        optimizer = Adam(lr=self.learning_rate, decay=self.decay)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    def fit(self, X, y, validation_data=None):
        """
        Train the neural networks.
        :param X: feature numpy array
        :param y: label numpy array
        :param validation_data: could be a tuple (x_val, y_val) or a tuple (x_val, y_val, val_sample_weights)
        :return: None
        """
        self.dim_inputs = X.shape[1]

        # construct model
        self._init_structure()
        self._init_learning()

        self.train_history = self.model.fit(x=X, y=y, batch_size=self.batch_size, epochs=self.epochs,
                                            verbose=self.verbose, validation_data=validation_data, shuffle=True)

    def predict(self, X):
        """
        Predict.
        :param X: feature numpy array
        :return: predicted y
        """
        return self.model.predict(x=X, batch_size=self.batch_size)

    def plot_train_history(self):
        """
        Plot the train loss history.
        :return: None
        """
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(self.train_history.history['loss'])
        plt.plot(self.train_history.history['val_loss'])
        plt.legend(['Training', 'Validation'])

        plt.show()


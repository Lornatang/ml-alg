import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(123)

X, y = make_blobs(centers=4, n_samples=5000)
# reshape targets to get column vector with shape (n_samples, 1)
y = y[:, np.newaxis]
# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)


class SoftmaxRegressor:
    def __init__(self):
        pass

    def train(self, X, y, n_classes, n_iters=10, learning_rate=0.1):
        """
        Trains a multinomial logistic regression model on given set of training data
        """
        self.n_samples, n_features = X.shape
        self.n_classes = n_classes
        self.weights = np.random.rand(self.n_classes, n_features)
        self.bias = np.zeros((1, self.n_classes))
        all_losses = []
        for i in range(n_iters):
            scores = self.compute_scores(X)
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y)
            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)
            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)
            self.weights = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db
            if i % 100 == 0:
                print(f"Iteration number: {i}, loss: {np.round(loss, 4)}")
        return self.weights, self.bias, all_losses

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            numpy array of shape (n_samples, 1) with predicted classes
        """
        scores = self.compute_scores(X)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def softmax(self, scores):
        """
        Tranforms matrix of predicted scores to matrix of probabilities
        Args:
            scores: numpy array of shape (n_samples, n_classes)
            with unnormalized scores
        Returns:
            softmax: numpy array of shape (n_samples, n_classes)
            with probabilities
        """
        exp = np.exp(scores)
        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = exp / sum_exp
        return softmax

    def compute_scores(self, X):
        """
        Computes class-scores for samples in X
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            scores: numpy array of shape (n_samples, n_classes)
        """
        return np.dot(X, self.weights.T) + self.bias

    def cross_entropy(self, y, scores):
        loss = - (1 / self.n_samples) * np.sum(y * np.log(scores))
        return loss

    def one_hot(self, y):
        """
        Tranforms vector y of labels to one-hot encoded matrix
        """
        one_hot = np.zeros((self.n_samples, self.n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot


regressor = SoftmaxRegressor()
w_trained, b_trained, loss = regressor.train(
    X_train, y_train, learning_rate=0.1, n_iters=1000, n_classes=4)
n_test_samples, _ = X_test.shape
y_predict = regressor.predict(X_test)
print(
    f"Classification accuracy on test set: {(np.sum(y_predict == y_test)/n_test_samples) * 100}%")

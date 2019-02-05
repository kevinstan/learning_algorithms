
# Author: Kevin Tan

import numpy as np
from sklearn import utils
from collections import Counter
from itertools import product
from collections import defaultdict

from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import mnist
x_train, y_train, x_test, y_test = mnist.load()

def random_prototype(x_train, y_train, M):
    x_train_M, y_train_M = utils.shuffle(x_train, y_train, n_samples=M)
    return (x_train_M, y_train_M)

def cnn(x_train, y_train, M):
    x_S, y_S = [], []
    x_S.append(x_train[0])
    y_S.append(y_train[0])
    x_train, y_train = x_train[1:,:], y_train[1:]
    knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
    while len(x_S) < M:
        if len(x_S) % 5 == 1: 
            knn.fit(x_S, y_S)
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            if knn.predict(x.reshape(-1, 784)) != y:
                x_S.append(x)
                y_S.append(y)
                mask = np.ones(len(x_train), dtype=bool)
                mask[i] = False
                x_train, y_train = x_train[mask,...], y_train[mask,...]
                break
    x_S, y_S = np.asarray(x_S), np.asarray(y_S)
    
    return (x_S, y_S)

def mean_and_std(rand_scores, cnn_scores):
    random_final = defaultdict(list)
    cnn_final = defaultdict(list)

    for M in rand_scores.keys():
        random_final[M].append(np.mean(rand_scores[M]))
        random_final[M].append(np.std(rand_scores[M]))

    for M in cnn_scores.keys():
        cnn_final[M].append(np.mean(cnn_scores[M]))
        cnn_final[M].append(np.std(cnn_scores[M]))

    return random_final, cnn_final

def main():

    M_list = [1000, 5000, 10000]
    n_exps = 8
    rand_scores = defaultdict(list)
    cnn_scores = defaultdict(list)

    for j in range(n_exps):
        for M in M_list:
            print('Training size:', x_train.shape)
            rand_pt = random_prototype(x_train, y_train, M)
            cnn_pt = cnn(x_train, y_train, M)
            print(rand_pt[0].shape, rand_pt[1].shape, cnn_pt[0].shape, cnn_pt[1].shape)

            rand_clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
            rand_clf.fit(rand_pt[0], rand_pt[1])
            cnn_clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
            cnn_clf.fit(cnn_pt[0], cnn_pt[1])

            rand_scores[M].append(rand_clf.score(x_test, y_test))
            cnn_scores[M].append(cnn_clf.score(x_test, y_test))
        print('Experiment number: ', j+1)
        print('Random scores:\n', rand_scores)
        print('CNN scores:\n', cnn_scores)

    print('Random test scores:\n', rand_scores)
    print('CNN test scores:\n', cnn_scores)

    random_final, cnn_final = mean_and_std(rand_scores, cnn_scores)
    print('Random scores mean and std:\n', random_final)
    print('CNN scores mean and std:\n', cnn_final)

if __name__ == '__main__':
    main()

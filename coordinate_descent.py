# Author: Kevin Tan

import numpy as np
import math

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
%config InlineBackend.figure_format = 'retina' 
%matplotlib inline
plt.rcParams['figure.figsize'] = 12,12

# scipy
import scipy.io as sio 
from scipy.optimize import line_search
from scipy.special import expit

# sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.utils import shuffle

### [--- Data Preprocessing ---] ###

# Load data
data = load_wine()
target = np.asarray(data.target)
data = np.asarray(data.data)
n = len(data)
X = np.asarray([data[i] for i in range(n) if target[i] != 2])
y = np.asarray([target[i] for i in range(n) if target[i] != 2])

# Convert class 0 to -1
y = np.asarray([-1 if y[i] == 0 else y[i] for i in range(len(y))])

# Normalize 
X = StandardScaler().fit_transform(X)

# Append bias term 
X = np.asarray(np.hstack((np.ones((len(X),1)), X)))

# Shuffle
X, y = shuffle(X, y, random_state=0)

### [--- Logistic Regression Baseline ---] ###

clf = LogisticRegression(random_state=0, C=10**10, fit_intercept=False, solver='liblinear')
clf.fit(X,y)
params = clf.get_params()
print('classes:', clf.classes_)
print('weights:', clf.coef_)
print('intercept:', clf.intercept_)

from sklearn.metrics import log_loss
L_star = log_loss(y, clf.predict_proba(X))
print("L* = ", L_star)

### [--- Coordinate Descent ---] ###

def sigmoid(x):
    return expit(x)

def L_w(X, y, w):
	log_losses = []
	for i in range(X.shape[0]):
	    dot = np.dot(w.T, X[i])
	    assert dot.shape[0]==1
	    exp_arg = -y[i] * dot
	    loss_i = np.log(1+np.exp(exp_arg))
	    log_losses.append(loss_i)
	return np.sum(log_losses)

def L_prime_w(X, y, w):
    losses = []
    for i in range(X.shape[0]):
        sig_arg = y[i]*np.dot(w.T, X[i].reshape(14,1))
        sig = sigmoid(-1.0 * sig_arg)
        loss_i = (-y[i] * sig) * X[i]
        losses.append(np.asarray(loss_i).reshape(14,))
    grad = np.sum(np.asarray(losses), axis=0)
    assert grad.shape[0]==14
    return grad

def random_idx(n):
    return np.random.randint(n)

def greedy_idx(gradient, n):
    return np.argmax(np.absolute(gradient))

def backtracking_line_search(X, y, w, beta=0.8):
    t = 1
    while True:
        grad = L_prime_w(X,y,w)
        lhs = L_w(X, y, w - t * grad.reshape(14,1))
        rhs = L_w(X, y, w) - (t * 1.0 / 2) * np.linalg.norm(grad, ord=2)**2
        if lhs <= rhs:
            break
        t = beta * t
    return t

def coordinate_descent(X, y, max_iter=4000, rand=False):
    # intialize w with some noise
    w = np.zeros((X.shape[1], 1)) 
    w += np.random.normal(0, 0.01, (X.shape[1],1))
    
    # track loss
    iterations = [0]
    L_w_list = [L_w(X, y, w)]
    
    for t in range(max_iter):
        # choose idx
        gradient = L_prime_w(X, y, w)
        if rand:
            idx = random_idx(X.shape[1])
        else:
            idx = greedy_idx(gradient, X.shape[1])
        gradient_j = np.asarray([gradient[idx] if j == idx else 0 for j in range(gradient.shape[0])])
        
        step_size = backtracking_line_search(X, y, w)
        
        # update scheme
        w_new = w - step_size * gradient_j.reshape(14,1)
        
        # track loss
        iterations.append(t+1)
        curr_loss = L_w(X, y, w_new)
        prev_loss = L_w(X, y, w)
        L_w_list.append(curr_loss)
        
        if t % 50 == 0:
            print('current loss at iteration {} is {}'.format(t, curr_loss))
        
        # terminate condition
        if np.linalg.norm(w_new - w, ord = 1) < 0.001:
            print("gradient descent has converged after " 
                  + str(t) + " iterations")
            break
    
        w = w_new
    
    return (w, iterations, L_w_list)
        
def plot(rand_iters, greedy_iters, rand_losses, greedy_losses):
    plt.title('Training curve')
    plt.xlabel('iteration')
    plt.ylabel('L(w)')
    plt.semilogy(rand_iters, np.array(rand_losses).reshape(-1, 1), 'r', label='test')
    plt.semilogy(greedy_iters, np.array(greedy_losses).reshape(-1, 1), 'b')
    plt.legend(('random feature', 'our algorithm'))
    plt.show()

def main():
	rand = coordinate_descent(X, y, rand=True)
	w_star, rand_iters, rand_losses = rand[0], rand[1], rand[2]
	print("optimal w vector with rand is:", str(w_star))
	print('last 5 losses:', rand_losses[-5:])

	greedy = coordinate_descent(X, y, rand=False)
	w_star, greedy_iters, greedy_losses = greedy[0], greedy[1], greedy[2]
	print("optimal w vector with greedy is:", str(w_star))
	print('last 5 losses:', greedy_losses[-5:])

	plot(rand_iters, greedy_iters, rand_losses, greedy_losses)

if __name__ == '__main__':
    main()

















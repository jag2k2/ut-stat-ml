import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from operator import lt,ge
from sklearn.model_selection import train_test_split

class DecisionStump:
    """
    A simple decision stump classifier
    dim : dimension on which to split
    value : value of the dimension
    op : comparator function (either <, <, >= or <=)
    """
    def __init__(self, dim=0, value=0, op=lt):
        self.dim = dim
        self.value = value
        self.op = op
        
    def update(self, dim=None, value=None, op=None):
        if dim is not None: self.dim = dim
        if value is not None: self.value = value
        if op is not None: self.op = op
    
    def predict(self,X):
        results = []
        for x in X[:, self.dim]:
            if self.op == '<':
                if x < self.value:
                    results.append(1)
                else:
                    results.append(-1)
            if self.op == '>':
                if x > self.value:
                    results.append(1)
                else:
                    results.append(-1)
        return np.array(results)        
        # return np.array([1 if self.op(x, self.value) else -1 for x in X[:,self.dim]])         
                
    """
    Fit a one-dimensional Decision Stump classifier.
    You should identify the dimension which results in the best split and find the corresponding optimal 
    feature threshold value.
    To facilitate the search for the optimal splitting dimension, this function is called by the fit_data 
    function for every dimension.
    """
    def fit_dim(self,X,Y,sample_weights,num_splits):
        """
        Input:
        X and Y are the input data and labels, respectively. 
        sample_weights are the iterated weights (initialized as (1/m,1/m,1/m,...,1/m))
        num_splits is the number of possible threshold values used for splitting the
        data (we discretize possible threshold values, find the optimal one); this value 
        should be less than the number of samples for the sake of computational efficiency.
        
        Return:
        min_err: the minimum classification error
        split_value: the optimal threshold 
        op: optimal operator (i.e., either >, <, >=, or <=)
        """
        m = len(X)
        min_err = float('inf')
        split_value = None
        op = None

        for value in np.linspace(min(X), max(X), num_splits):
            for operator in ['>', '<']:
                incorrect = 0
                for i in range(m):
                    if (operator == '>' and X[i] > value) or (operator == '<' and X[i] < value):
                        prediction = 1 if Y[i] == 1 else -1
                    else:
                        prediction = 1 if Y[i] == -1 else -1
                    if prediction != 1:
                        incorrect += sample_weights[i]

                if incorrect < min_err:
                    min_err = incorrect
                    split_value = value
                    op = operator

        return min_err, split_value, op
    
    """
    Finding an optimal splitting dimension and the corresponding feature threshold value
    X : n x d data matrix, n number of samples with d dimension
    Y : n dimensional array containing label of each observation, label = {-1,1}
    sample_weights : weight of each observation
    num_splits : number of split value to be tested randomly
    
    """
    def fit_data(self,X,Y,sample_weights,num_splits=100):
        """
        Input:
        X and Y are the input data and labels, respectively.
        sample_weights are the iterated weights (initialized as (1/m,1/m,1/m,...,1/m))
        num_splits is the number of possible threshold values used for splitting the
        data (we discretize possible threshold values, find the optimal one); this value 
        should be less than the number of samples for the sake of computational efficiency.
        There is no return in this funtion
        you can use update() to assign the optimal value of self.dim, self.value and self.op
        """
        m, d = X.shape
        min_err = float('inf')
        best_dim = None
        best_value = None
        best_op = None

        for dim in range(d):
            X_dim = X[:, dim]
            err, value, op = self.fit_dim(X_dim, Y, sample_weights, num_splits)
            if err < min_err:
                min_err = err
                best_dim = dim
                best_value = value
                best_op = op

        self.dim = best_dim
        self.value = best_value
        self.op = best_op

class Adaboost:
    def __init__(self, n):
        self.weak_learners = []
        self.learner_weights = []
        self.sample_weights = np.repeat(1/n,n)    # to begin all samples get the same weight
                  
    
    def add_learner(self, X, Y, weak_learner_class = DecisionStump):
        """
        In this function, Adaboost completes one iteration
        Please use the class DecisionStump and its member functions you definted above.
        
        X : n x d data matrix, n is the number of samples in d dimensions
        Y : n dimensional array containing label of each observation, label in {-1,1}
        weak_learner_class is default: DecisionStump
        Notice: you need to use a list to append all the weak learner objects and their weights here
        """
        # Train the weak learner
        weak_learner = weak_learner_class()
        weak_learner.fit_data(X, Y, self.sample_weights)
        
        # Compute the error of the weak learner
        predictions = weak_learner.predict(X)
        error = np.sum(self.sample_weights * (predictions != Y))
        print("error", np.sum(predictions != Y))

        # Compute the weight of the weak learner
        alpha = 0.5 * np.log((1 - error) / error)
        self.learner_weights.append(alpha)
        
        # Update the sample weights
        self.sample_weights *= np.exp(-alpha * Y * predictions)
        self.sample_weights /= np.sum(self.sample_weights)
        
        # Append the weak learner to the list
        self.weak_learners.append(weak_learner)
    
    def predict(self,X):
        """
        You can use this function to return predicted label using the iterated strong model.
        The current iterated strong model H(t) consists of t weak learners we saved before 
        """
        predictions = np.zeros(X.shape[0])
        for learner, weight in zip(self.weak_learners, self.learner_weights):
            predictions += weight * learner.predict(X)
        return np.sign(predictions)
        #return predictions
    
    def prediction_error(self,X,Y):
        """
        You can use this function to return the predicted error using the iterated strong model
        """
        predictions = self.predict(X)
        error = np.mean(predictions != Y)
        return error
    

def plot_results(train_error, test_error):
    """
    Plot error on the training and test set as a function 
    of the number of rounds of boosting.
    """
    x = np.arange(len(train_error))
    plt.figure(figsize=(10,6))
    plt.plot(x, train_error, label='train error', marker='o')
    plt.plot(x, test_error, label='test error', marker='o')
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('error')
    plt.title('training and test set error')
    plt.show()
#Read in the dataset
dataFile = 'spamdata.mat'
data = scipy.io.loadmat(dataFile)['spamdata']
X = data[:,0:57]
y = data[:,57]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=50)

# normalize data
X_train = np.log(X_train + 0.1 * np.ones((len(X_train), len(X_train[0]))))
X_test = np.log(X_test + 0.1 * np.ones((len(X_test), len(X_test[0]))))
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

#Maximum number of weak learners to be used in Adaboost
max_num_weak_learners = 40

#Train and test error
train_error = []
test_error = []

#Training Adaboost with weak learners
n,d = X_train.shape
model = Adaboost(n)
for m in range(1, max_num_weak_learners + 1):
    print("Training Adaboost with weak learners %d" % m)
    model.add_learner(X_train, y_train)
    train_error.append(model.prediction_error(X_train, y_train))
    test_error.append(model.prediction_error(X_test, y_test))

print("Initial Training Error=%.4f Testing Error= %.4f " % (train_error[0], test_error[0]))
print("Final Training Error=%.4f Testing Error= %.4f " % (train_error[-1], test_error[-1]))
plot_results(train_error, test_error)
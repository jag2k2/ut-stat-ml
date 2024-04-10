import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from operator import lt,ge
from sklearn.model_selection import train_test_split

class DecisionStump:
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
                
    def fit_dim(self,X,Y,sample_weights,num_splits):
        num_rows = len(X)
        min_err = float('inf')
        split_value = None
        op = None

        for threshold_value in np.linspace(min(X), max(X), num_splits):
            for operator in ['>', '<']:
                error = 0
                for i in range(num_rows):
                    if (operator == '>' and X[i] > threshold_value) or (operator == '<' and X[i] < threshold_value):
                        if Y[i] == 1:
                            prediction = 1
                        else:
                            prediction = -1
                    else:
                        if Y[i] == -1:
                            prediction = 1
                        else:
                            prediction = -1
                    if prediction != 1:
                        error += sample_weights[i]

                if error < min_err:
                    min_err = error
                    split_value = threshold_value
                    op = operator

        return min_err, split_value, op
    
    def fit_data(self,X,Y,sample_weights,num_splits=100):
        d = X.shape[1]
        min_err = float('inf')
        optimal_dim = None
        optimal_value = None
        optimal_op = None

        for dim in range(d):
            X_dim = X[:, dim]
            err, value, op = self.fit_dim(X_dim, Y, sample_weights, num_splits)
            if err < min_err:
                min_err = err
                optimal_dim = dim
                optimal_value = value
                optimal_op = op
        
        self.update(optimal_dim, optimal_value, optimal_op)

class Adaboost:
    def __init__(self, n):
        self.weak_learners = []
        self.learner_weights = []
        self.sample_weights = np.repeat(1/n,n)    # all samples weighted equally at the beginning
                  
    
    def add_learner(self, X, Y, weak_learner_class = DecisionStump):
        weak_learner = weak_learner_class()
        weak_learner.fit_data(X, Y, self.sample_weights)            # Invoke the weak learner
        pred_y = weak_learner.predict(X)
        error = np.sum(self.sample_weights * (pred_y != Y))         # Compute et
        alpha = 0.5 * np.log((1 - error) / error)                   # Calculate learner weight
        self.learner_weights.append(alpha)                          # Add the weak learner weights to the list
        self.sample_weights *= np.exp(-alpha * Y * pred_y)          
        self.sample_weights /= np.sum(self.sample_weights)          # Update sample weights
        self.weak_learners.append(weak_learner)                     # Add the weak learner object to the list
    
    def predict(self,X):
        predictions = np.zeros(X.shape[0])
        for weak_learner, learner_weight in zip(self.weak_learners, self.learner_weights):
            predictions += learner_weight * weak_learner.predict(X)
        return np.sign(predictions)
    
    def prediction_error(self,X,Y):
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
    plt.plot(x, train_error, label='Training Set Error', marker='o')
    plt.plot(x, test_error, label='Test Set Error', marker='o')
    plt.legend()
    plt.xlabel('Rounds of Boosting')
    plt.ylabel('Error')
    plt.title('Training and Test Set Error')
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
max_num_weak_learners = 10

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
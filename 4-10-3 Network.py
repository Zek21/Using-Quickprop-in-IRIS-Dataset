import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''a 4-10-3 network,'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Load the data into a pandas dataframe
iris_df = pd.read_csv(url, header=None)

# Extract the features and target variable
X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, -1].values

# Convert the target variable to one-hot encoded vectors
y_one_hot = np.zeros((y.shape[0], 3))
for i in range(y.shape[0]):
    if y[i] == 'Iris-setosa':
        y_one_hot[i, 0] = 1
    elif y[i] == 'Iris-versicolor':
        y_one_hot[i, 1] = 1
    else:
        y_one_hot[i, 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=4)

learning_rate = 0.1
iterations = 1000
N = y_train.shape[0]

# Input features
input_size = 4

# Hidden layer
hidden_size = 10

# Output layer
output_size = 3

results = pd.DataFrame(columns=["mse", "accuracy"])

np.random.seed(10)

# Hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

# Output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2 * y_pred.shape[0])

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

# Initialize previous weight changes for Quickprop
dW1_prev = np.zeros_like(W1)
dW2_prev = np.zeros_like(W2)

# Quickprop constants
mu = 1.75
min_update = 1e-6

for itr in range(iterations):
    # Implementing feedforward propagation on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)

    # Implementing feedforward propagation on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Calculating the error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results = results.append({"mse": mse, "accuracy": acc}, ignore_index=True)

    # Backpropagation phase
    E2 = (A2 - y_train) * A2 * (1 - A2)
    dW2 = np.dot(A1.T, E2) / N

    E1 = np.dot(E2, W2.T) * A1 * (1 - A1)
    dW1 = np.dot(X_train.T, E1) / N

    # Quickprop updates
    W1_delta = learning_rate * dW1
    W2_delta = learning_rate * dW2

    W1 += W1_delta
    W2 += W2_delta

    # Compute change in weights
    dW1_change = dW1 - dW1_prev
    dW2_change = dW2 - dW2_prev

    # Compute weight update ratios with Quickprop
    quickprop_eta = np.where(dW1_change != 0, W1_delta / (dW1_change + min_update), 1.0)
    quickprop_eta = np.minimum(quickprop_eta, mu)

    quickprop_eta2 = np.where(dW2_change != 0, W2_delta / (dW2_change + min_update), 1.0)
    quickprop_eta2 = np.minimum(quickprop_eta2, mu)

    # Update weights with Quickprop
    W1 -= quickprop_eta * dW1_change
    W2 -= quickprop_eta2 * dW2_change

    # Store current weight changes for the next iteration
    dW1_prev = dW1.copy()
    dW2_prev = dW2.copy()

plt.plot(results.mse, label='Training Loss')
plt.plot(results.accuracy, label='Testing Loss')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("Accuracy:", (acc))
print("Training Loss:", (mse))


def predict(X, W1, W2):
    """
    Makes predictions using a trained neural network model.
    
    Arguments:
    X -- input data (n_samples, n_features)
    W1 -- weights of the hidden layer (n_features, n_hidden_units)
    W2 -- weights of the output layer (n_hidden_units, n_classes)
    
    Returns:
    predictions -- predicted class labels (n_samples, )
    """
    # Implementing feedforward propagation on hidden layer
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    # Implementing feedforward propagation on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Get the index of the maximum value in each row
    predictions = np.argmax(A2, axis=1)
    
    label_to_name = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    flower_names = np.array([label_to_name[label] for label in predictions])
    
    return predictions, flower_names

# Load new data
new_data = np.array([[5.3, 3.7, 1.5, 0.2], [7.0, 3.2, 4.7, 1.4]])

# Make predictions on new data
predictions = predict(new_data, W1, W2)

# Print the predicted class labels
print(predictions)

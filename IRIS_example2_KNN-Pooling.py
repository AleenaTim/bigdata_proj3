import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import multiprocessing as mp

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a function to train a K-NN model
def train_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return (k, accuracy)

# Define the list of k values for the K-NN classifiers
k_values = [3, 5, 7]

# Use multiprocessing to train the K-NN classifiers in parallel
if __name__ == '__main__':
    with mp.Pool(processes=3) as pool:
        results = pool.map(train_knn, k_values)

    # Print the results
    for k, accuracy in results:
        print(f"K-NN with k={k}: Accuracy = {accuracy:.2f}")

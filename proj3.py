import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from scipy.stats import mode
from multiprocessing import Pool
import time

# 1. Read the CSV file
file_path = 'breast-cancer.csv'
df = pd.read_csv(file_path)

# 2. Data Cleaning / Preparation
df_clean = df.dropna()  # Remove rows with any missing values

# 3. Split dataset into features (X) and target (y)
# 'diagnosis' is assumed to be the target variable
X = df_clean.drop(columns=['diagnosis'])  # Features
y = df_clean['diagnosis']  # Target variable

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the SVM with RBF kernel
start_time_rbf = time.time()
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
training_time_rbf = time.time() - start_time_rbf

# Make predictions
y_pred_rbf = svm_rbf.predict(X_test)

# Evaluate the RBF SVM model
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
sensitivity_rbf = recall_score(y_test, y_pred_rbf, pos_label='M')  # Adjust pos_label based on your dataset
specificity_rbf = recall_score(y_test, y_pred_rbf, pos_label='B')  # Adjust pos_label based on your dataset

print(f"Training time (RBF): {training_time_rbf:.2f} seconds")
print(f"Accuracy (RBF): {accuracy_rbf:.2f}")
print(f"Sensitivity (RBF): {sensitivity_rbf:.2f}")
print(f"Specificity (RBF): {specificity_rbf:.2f}")

# Visualize the confusion matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf, labels=['M', 'B'])  # Adjust labels based on your dataset
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=['M', 'B'], yticklabels=['M', 'B'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (RBF Kernel)')
plt.show()

# (a) Visualize the class boundary
# Select two features for visualization
feature1, feature2 = X.columns[0], X.columns[1]  # Example: first two features
X_vis = X[[feature1, feature2]]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.2, random_state=42)

# Train SVM on selected features
svm_rbf_vis = SVC(kernel='rbf')
svm_rbf_vis.fit(X_train_vis, y_train_vis)

# Create mesh grid for plotting
x_min, x_max = X_vis[feature1].min() - 1, X_vis[feature1].max() + 1
y_min, y_max = X_vis[feature2].min() - 1, X_vis[feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict on mesh grid
Z = svm_rbf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_vis[feature1], X_vis[feature2], c=y, edgecolor='k', marker='o')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('SVM Decision Boundary (RBF Kernel)')
plt.show()

# 6. Train three SVM classifiers with different kernels using multiprocessing
def train_svm(kernel):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

# Start the timer for ensemble
start_time_ensemble = time.time()

# Train classifiers in parallel
kernels = ['linear', 'rbf', 'poly']
with Pool(processes=3) as pool:
    models = pool.map(train_svm, kernels)

training_time_ensemble = time.time() - start_time_ensemble

# Make predictions from each classifier
predictions = [model.predict(X_test) for model in models]

# Majority voting for ensemble
stacked_predictions = np.stack(predictions, axis=1)
majority_votes = mode(stacked_predictions, axis=1)[0]

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, majority_votes)
sensitivity_ensemble = recall_score(y_test, majority_votes, pos_label='M')  # Adjust pos_label based on your dataset
specificity_ensemble = recall_score(y_test, majority_votes, pos_label='B')  # Adjust pos_label based on your dataset

print(f"Training time (Ensemble): {training_time_ensemble:.2f} seconds")
print(f"Accuracy (Ensemble): {accuracy_ensemble:.2f}")
print(f"Sensitivity (Ensemble): {sensitivity_ensemble:.2f}")
print(f"Specificity (Ensemble): {specificity_ensemble:.2f}")

# Visualize confusion matrix for ensemble
cm_ensemble = confusion_matrix(y_test, majority_votes, labels=['M', 'B'])  # Adjust labels based on your dataset
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=['M', 'B'], yticklabels=['M', 'B'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Ensemble)')
plt.show()

# Compare training times
print(f"Training time for RBF SVM: {training_time_rbf:.2f} seconds")
print(f"Training time for Ensemble: {training_time_ensemble:.2f} seconds")
Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

step 2: Load and Prepare Dataset

from google.colab import files
uploaded = files.upload()  # This will open a file picker to upload from your local machine
Output:
breast-cancer.csv(text/csv) - 124571 bytes, last modified: 6/5/2025 - 100% done
Saving breast-cancer.csv to breast-cancer.csv

import pandas as pd
df = pd.read_csv("breast-cancer.csv")

Step 3: Select 2 Features for 2D Visualization

features = ['radius_mean', 'texture_mean']
X = df[features].values
y = df['diagnosis'].values

Step 4: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 5: Standardize the Data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Step 6: Train SVM with Linear Kernel

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_scaled, y_train)

Output:
SVC(C=1, kernel='linear')

Step 7: Train SVM with RBF Kernel

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

output:
SVC(C=1)

Step 8: Visualize Decision Boundaries

def plot_decision_boundary(clf, X, y, scaler, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Stack grid points and scale them before prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    
    Z = clf.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

      
 step 9: Hyperparameter Tuning with Grid Search

  param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

output:

GridSearchCV(cv=5, estimator=SVC(),
             param_grid={'C': [0.1, 1, 10, 100],
                         'gamma': ['scale', 0.01, 0.1, 1, 10],
                         'kernel': ['rbf']})

best_estimator_: SVC
SVC(C=10, gamma=0.1)

SVC
?
SVC(C=10, gamma=0.1)

Step 10: Evaluate with Cross-Validation and Test Accuracy

best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("CV Accuracy: {:.2f}% ± {:.2f}%".format(cv_scores.mean() * 100, cv_scores.std() * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

Output:
Best Parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
CV Accuracy: 90.11% ± 3.11%
Test Accuracy: 91.23%




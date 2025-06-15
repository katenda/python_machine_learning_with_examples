import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# 1. Load Dataset and Inspect
dataset = pd.read_csv('data/social_network_ads.csv')

# --- DIAGNOSTIC STEP ---
# Let's check for missing values in any column
print("Checking for missing values:")
print(dataset.isnull().sum())
# --- END DIAGNOSTIC STEP ---

# Let's drop rows with any missing values for now.
# This is a simple strategy. More advanced strategies involve 'imputation' (filling in values).
dataset.dropna(inplace=True)

X = dataset.iloc[:, [2, 3]].values  # Age and EstimatedSalary
y = dataset.iloc[:, -1].values     # Purchased

# 2. Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3. Feature Scaling
# This is crucial for performance. We scale the data so that all features have a similar range.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Train the Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 5. Predict the test set results
y_pred = classifier.predict(X_test)

# 6. Evaluate the model
# The Confusion Matrix shows us the number of correct and incorrect predictions.
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Accuracy is the percentage of correct predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 7. Visualize the results (Decision Boundary)
# This is a complex but powerful visualization that shows how the model separates the data.
def plot_decision_boundary(X_set, y_set, set_name):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=[ListedColormap(('red', 'green'))(i)], label=j)
        
    plt.title(f'Logistic Regression ({set_name})')
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.show()

# Plotting for the Training set
plot_decision_boundary(X_train, y_train, 'Training set')

# Plotting for the Test set
plot_decision_boundary(X_test, y_test, 'Test set') 
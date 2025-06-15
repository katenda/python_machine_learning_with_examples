import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 1. Load and Preprocess Data
dataset = pd.read_csv('data/credit_card_default.csv')

# Drop the customer ID as it's not a predictive feature
dataset = dataset.drop('CustomerID', axis=1)

# Handle missing values if any (using a simple drop for this example)
dataset.dropna(inplace=True)

# --- One-Hot Encoding for Categorical Features ---
# This is the key step to convert text data into a format the model can use.
dataset = pd.get_dummies(dataset, columns=['Gender', 'Education'], drop_first=True)

# 2. Define Features (X) and Target (y)
# All columns except 'Default' are features
X = dataset.drop('Default', axis=1)
y = dataset['Default']

# 3. Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Note: 'stratify=y' is important for imbalanced datasets. It ensures that the train and test sets
# have the same proportion of target classes as the original dataset.

# 4. Feature Scaling
# We only scale the columns that are not one-hot encoded (i.e., the original numerical ones)
numerical_cols = ['Age', 'Income', 'LoanAmount']
scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 5. Train the Logistic Regression model
# We can add class_weight='balanced' to help with the imbalanced data
classifier = LogisticRegression(random_state=42, class_weight='balanced')
classifier.fit(X_train, y_train)

# 6. Predict the test set results
y_pred = classifier.predict(X_test)

# 7. Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 
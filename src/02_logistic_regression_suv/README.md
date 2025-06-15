# Logistic Regression Example 1: Predicting SUV Purchase

## 1. The Business Problem

A car company has released a new SUV and wants to run a targeted marketing campaign. Instead of showing ads to everyone, they want to focus on customers who are most likely to purchase the vehicle. They have a dataset of their existing customers, including their age, estimated salary, and whether they have purchased a similar product in the past.

**Our Goal:** Build a model that can predict whether a given customer will purchase the new SUV based on their age and salary.

## 2. The Dataset (`social_network_ads.csv`)

*   **Features (Inputs):**
    *   `Age`: The customer's age.
    *   `EstimatedSalary`: The customer's estimated salary.
*   **Target (Output):**
    *   `Purchased`: A binary value. `1` if they purchased, `0` if they did not.

## 3. The Solution: `logistic_regression_1.py`

This script builds our prediction model. Here is a breakdown of the key steps.

### Step 1: Data Loading and Cleaning

We first load the `.csv` file using pandas. Crucially, we immediately check for missing values using `dataset.isnull().sum()`. In our case, we found one row with missing data. For this example, we used `dataset.dropna(inplace=True)` to simply remove that row. This is a vital first step in any real-world project.

### Step 2: Feature Selection and Data Splitting

We select our input features (`X`), which are 'Age' and 'EstimatedSalary', and our output target (`y`), which is 'Purchased'. We then split our data into a training set (for the model to learn from) and a test set (to evaluate its performance on unseen data).

### Step 3: Feature Scaling (A Critical Concept)

This is one of the most important steps. The `Age` (e.g., 20-60) and `EstimatedSalary` (e.g., 20,000-150,000) features are on vastly different scales. Many ML algorithms can be biased towards features with larger values.

We use `StandardScaler` to transform both features so they are on a similar scale. This prevents the salary from unfairly dominating the model's decision-making process just because its numbers are bigger.

### Step 4: Training the Model

We train a `LogisticRegression` model from the scikit-learn library using our scaled training data (`X_train`, `y_train`).

### Step 5: Evaluation

This is how we judge our model's performance.

*   **Accuracy:** We achieved **91% accuracy** on the test set, meaning we correctly predicted the outcome for 91% of the customers the model had never seen before.
*   **Confusion Matrix:** This provides a deeper look:
    *   **True Negatives (24):** Correctly predicted 24 people would NOT buy.
    *   **True Positives (18):** Correctly predicted 18 people WOULD buy.
    *   **False Positives (1):** Incorrectly predicted 1 person would buy (a "mistake").
    *   **False Negatives (3):** Incorrectly predicted 3 people would NOT buy, but they actually would have (a "missed opportunity").

### Step 6: Visualization

The script generates a plot showing the **decision boundary**.
*   The red area represents the region where the model predicts 'Will Not Buy'.
*   The green area represents the region where the model predicts 'Will Buy'.
*   The dots are the actual customers from our test set.
*   This plot visually confirms that our model has learned a logical boundary: older customers with higher salaries are classified as likely buyers.

## 4. Interview Corner

**Q1: "You used Feature Scaling. Why was it necessary?"**
> **A:** "The 'Age' and 'EstimatedSalary' features had very different scales. To prevent the algorithm from giving more weight to 'EstimatedSalary' simply because its numerical values were larger, I used `StandardScaler`. This standardizes the features to a common scale, ensuring the model learns the true underlying patterns without being biased by the magnitude of the numbers."

**Q2: "Your model had 3 'False Negatives'. What does that mean in this business context?"**
> **A:** "A False Negative means our model predicted a customer would *not* buy, but they actually *would* have. In this context, that represents 3 missed opportunities. These are potential sales we would not have captured with our targeted ad campaign. Depending on the profit margin of the SUV, these missed opportunities could be very costly, and it might be a business priority to tune the model to reduce them, even if it means slightly lowering the overall accuracy."

**Q3: "Could you use Linear Regression for this problem? Why or why not?"**
> **A:** "No, Linear Regression is not suitable here. Linear Regression is designed to predict a continuous value (like a price or temperature). This is a classification problem where the output is a discrete category: 'buy' (1) or 'not buy' (0). Logistic Regression is the appropriate choice because it's designed for classification. Its output is a probability between 0 and 1, which can be mapped to our two categories." 
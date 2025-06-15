# Linear Regression Example: Predicting Student Scores

## 1. The Business Problem

A tutoring service wants to understand the relationship between the number of hours a student studies and the percentage score they receive on an exam. They want to be able to predict a student's score based on their study time.

**Our Goal:** Build a simple model to predict a student's exam score based on the number of hours they study.

## 2. The Dataset (`student_scores.csv`)

*   **Feature (Input):**
    *   `Hours`: The number of hours the student studied.
*   **Target (Output):**
    *   `Scores`: The percentage score the student achieved.

## 3. The Solution: `linear_regression.py`

This script builds our prediction model. Here is a breakdown of the key steps.

### Step 1: Data Loading and Splitting

We load the data and split it into a training set and a test set. This allows us to train the model on one portion of the data and test its performance on another, unseen portion.

### Step 2: Training the Model

We use the `LinearRegression` model from scikit-learn. The `.fit()` method is where the model "learns" the relationship between `Hours` and `Scores` from the training data. It calculates the best-fit straight line that describes the data.

### Step 3: Making Predictions

We use the trained model to predict the scores for the test set hours. The script also makes a specific prediction for a new data point (9.25 hours) to demonstrate how to use the model for a single new input.

### Step 4: Visualization

The script generates two plots:
1.  **Training Set:** Shows the original training data points (in red) and the linear regression line (in blue) that the model learned.
2.  **Test Set:** Shows the test data points and the same regression line. Seeing that the line fits the test data well gives us confidence that our model can generalize to new, unseen data.

## 4. Interview Corner

**Q1: "What does the blue line in your plot represent?"**
> **A:** "The blue line is the linear regression model itself. It represents the relationship the model has learned between the number of hours studied and the predicted exam score. The equation of this line is `y = mx + c`, where `y` is the predicted score, `x` is the number of hours, `m` is the slope (how much the score increases for each extra hour of study), and `c` is the y-intercept (the predicted score for 0 hours of study)."

**Q2: "What are some limitations of Linear Regression?"**
> **A:** "Linear Regression's primary limitation is in its name: it only works well when the underlying relationship between the features and the target is linear. If the relationship is curved or more complex, a linear model will not be accurate. It's also sensitive to outliers, as a single extreme data point can significantly skew the position of the regression line." 
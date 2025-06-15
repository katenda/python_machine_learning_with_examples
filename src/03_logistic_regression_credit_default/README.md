# Logistic Regression Example 2: Predicting Credit Card Default

## 1. The Business Problem

A bank wants to proactively identify customers who are at high risk of defaulting on their credit card payments. By predicting potential defaulters, the bank can take preventative measures, such as offering financial counseling or adjusting credit limits, to mitigate losses.

**Our Goal:** Build a classification model to predict whether a customer will default (`1`) or not (`0`) based on their demographic and financial data.

## 2. The Dataset (`credit_card_default.csv`)

This dataset contains customer information.

*   **Features (Inputs):**
    *   `Gender`: Male or Female.
    *   `Age`: Customer's age.
    *   `Income`: Customer's annual income.
    *   `LoanAmount`: The amount of the loan on the credit card.
    *   `Education`: The customer's education level (e.g., 'High School', 'University').
*   **Target (Output):**
    *   `Default`: `1` if the customer defaulted, `0` otherwise.

## 3. The Solution: Key Concepts & Code

This example introduces two critical, real-world data science challenges.

### New Concept 1: Handling Categorical Data

Our dataset contains features with text values ('Gender', 'Education'). Machine learning models require all inputs to be numeric. We cannot simply ignore this data, as a customer's education level could be a very important predictor.

*   **The Solution:** We use a technique called **One-Hot Encoding**. The `pandas.get_dummies()` function will transform our categorical columns into new numerical columns. For example, the 'Education' column will be replaced by new columns like `Education_High School` and `Education_University`. A customer with a 'University' education will have a `1` in the `Education_University` column and a `0` in the others. This allows the model to understand the categorical information.

### New Concept 2: Imbalanced Data and Evaluation Metrics

In this dataset, there are many more non-defaulters (`0`) than defaulters (`1`). This is called an **imbalanced dataset**.

*   **The Problem:** If only 5% of customers default, a naive model that *always* predicts "no default" will be 95% accurate! This accuracy score is dangerously misleading. We need better metrics.
*   **The Solution:** We must look at the **Confusion Matrix** more closely and use **Precision** and **Recall**:
    *   **Precision:** Of all the customers the model *predicted* would default, what percentage actually did? (Measures the quality of the positive predictions).
    *   **Recall (Sensitivity):** Of all the customers who *actually* defaulted, what percentage did our model correctly identify? (Measures the ability to find all the positive cases).

For a bank, **Recall is often the most important metric**. They want to identify as many of the *actual* defaulters as possible, even if it means incorrectly flagging a few non-defaulters along the way. Missing a defaulter (a False Negative) is usually more costly than investigating a non-defaulter (a False Positive).

## 4. Interview Corner

**Q1: "Your dataset contained categorical features like 'Education'. How did you handle them and why?"**
> **A:** "The model requires numeric inputs, so I used one-hot encoding via `pandas.get_dummies()`. This technique creates new binary columns for each category (e.g., 'Education_University', 'Education_High School'), converting the non-numeric data into a format the model can process without implying an incorrect ordinal relationship between the categories."

**Q2: "The problem involves predicting 'default', which is often a rare event. How does this affect your choice of evaluation metric?"**
> **A:** "This is a classic imbalanced data problem, which makes accuracy a misleading metric. A model could achieve 95% accuracy by simply always predicting 'no default'. Therefore, I would focus on Precision and, more importantly, Recall. For a bank, Recall is critical—it tells us what percentage of the *actual* defaulters we successfully caught. A low recall means we are failing to identify at-risk customers, which could lead to significant financial losses. The goal would be to maximize Recall, possibly at the expense of some Precision."

**Q3: "What's the difference between a False Positive and a False Negative in this specific banking context?"**
> **A:**
> *   A **False Positive** is when the model predicts a customer *will* default, but they actually don't. The business cost is the effort spent on unnecessary intervention for a good customer (e.g., offering them counseling they don't need).
> *   A **False Negative** is when the model predicts a customer will *not* default, but they actually do. The business cost is a financial loss from an unpredicted default. In most banking scenarios, the cost of a False Negative is much higher than the cost of a False Positive." 
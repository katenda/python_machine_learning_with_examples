# Machine Learning Practice by Katenda Enock

This repository is a collection of hands-on examples to learn and practice machine learning concepts. Each example is designed to be a practical, self-contained introduction to a specific algorithm or technique.

See the `ROADMAP.md` file for a detailed, step-by-step plan of the topics we will cover.

## Project Structure

The project is organized into subdirectories within `src/`. Each subdirectory contains a self-contained example with its own code and a detailed `README.md` explaining the problem and solution.

```
.
├── data/
│   ├── student_scores.csv
│   └── social_network_ads.csv
├── src/
│   ├── 01_linear_regression/
│   │   ├── README.md
│   │   └── linear_regression.py
│   └── 02_logistic_regression_suv/
│       ├── README.md
│       └── logistic_regression_1.py
├── venv/
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## Getting Started

Follow these steps to set up your environment and run the examples.

### 1. Setup Virtual Environment

It is recommended to use a virtual environment to manage project dependencies. If you don't have one set up, you can create it with:

```bash
python3 -m venv venv
```

And activate it:

**On macOS and Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
.\\venv\\Scripts\\activate
```

### 2. Install Dependencies

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Running the Examples

All examples are located in subdirectories within the `src/` directory. To run an example, point Python to the correct file path.

#### Example 1: Linear Regression

```bash
python src/01_linear_regression/linear_regression.py
```

#### Example 2: Logistic Regression (SUV Purchase)

```bash
python src/02_logistic_regression_suv/logistic_regression_1.py
``` 
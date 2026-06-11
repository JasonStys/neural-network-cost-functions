# Neural Network Cost Functions

This repository contains a Python notebook from CS478 that implements and compares common cost functions used in binary classification and introductory neural network training.

The project focuses on squared error, mean quadratic cost, binary cross entropy, and mean cross entropy. It uses small prediction examples to show how each cost function responds when a model prediction is correct, close, uncertain, or confidently wrong.

## Project Overview

Cost functions measure how far a model prediction is from the true label. They are a core part of machine learning because training algorithms use them to guide model updates.

This notebook builds each cost function directly with NumPy, then evaluates the functions with example values. The examples make it easier to see why cross entropy is often preferred for binary classification. Cross entropy gives a much larger penalty when a model is confidently wrong, while squared error changes more gradually.

## Cost Functions Included

### Squared Error

The squared error function calculates the squared difference between a true value and a predicted value.

```python
def squared_error(y, yhat):
    return (y - yhat) ** 2
```

### Mean Quadratic Cost

The mean quadratic cost function averages squared errors across multiple examples.

```python
def mean_quadratic_cost(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean((y - yhat) ** 2)
```

### Binary Cross Entropy

The binary cross entropy function measures prediction error for binary labels. The function clips probabilities to avoid taking the logarithm of zero.

```python
def cross_entropy(y, a, eps=1e-15):
    a = np.clip(a, eps, 1 - eps)
    return -(y * log(a) + (1 - y) * log(1 - a))
```

### Mean Cross Entropy

The mean cross entropy function averages binary cross entropy across multiple examples.

```python
def mean_cross_entropy(y, a, eps=1e-15):
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    a = np.clip(a, eps, 1 - eps)
    return np.mean(-(y * log(a) + (1 - y) * log(1 - a)))
```

## Example Results

The notebook evaluates several binary classification examples.

For squared error, a correct prediction such as `y = 1` and `yhat = 1` produces `0`. A poor prediction such as `y = 1` and `yhat = 0.1192` produces a much larger error of about `0.775809`.

For cross entropy, a confident correct prediction such as `y = 1` and `a = 0.9997` produces a very small cost of about `0.0003`. A confident wrong prediction such as `y = 1` and `a = 0.01` produces a much larger cost of about `4.6052`.

The notebook also computes these aggregate values:

```text
Mean quadratic cost: 0.160002895
Mean cross entropy: 0.4995480159885067
```

## Technologies Used

- Python
- NumPy
- pandas
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys,_CS478_01,_CA4_cost_function.ipynb
├── Jason Stys_CS478-01_CA4_cost function.pdf
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/neural-network-cost-functions.git
cd neural-network-cost-functions
```

Install the required packages.

```bash
pip install numpy pandas notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- Python programming
- NumPy based mathematical computation
- pandas DataFrame creation and display
- Binary classification loss function implementation
- Manual verification of neural network cost behavior
- Jupyter Notebook workflow
- Google Colab workflow

## Possible Future Improvements

- Add plots comparing squared error and cross entropy
- Add gradients for each cost function
- Add explanations of how each function affects gradient descent
- Convert the notebook into a small Python module with unit tests
- Add examples using logistic regression predictions

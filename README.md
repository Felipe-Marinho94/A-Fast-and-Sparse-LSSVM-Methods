# Sparse Training Algorithms for Least Squares Support Vector Machines (LSSVM)

Implementation of efficient training algorithms for Least Squares Support Vector Machines (LSSVM) proposed in the PhD thesis:

**"Sparse and Efficient Dual Approaches for Least Squares Support Vector Machines"**  
Felipe Pinto Marinho

The repository implements novel optimization strategies for sparse and scalable training of LSSVM models, focusing on improved computational efficiency and model sparsity.

---

## Overview

Least Squares Support Vector Machines (LSSVM) are powerful kernel-based learning models derived from the classical Support Vector Machines (SVM). However, traditional LSSVM training requires solving a large linear system, which limits scalability for large datasets.

This project introduces new optimization strategies that improve training efficiency and allow the construction of sparse solutions.

The main contributions implemented in this repository are:

- **TCSMO-LSSVM**  
  A Three-Term Sequential Minimal Optimization method for efficient training of LSSVM.

- **SCG-LSSVM**  
  A Spectral Conjugate Gradient approach for solving the dual optimization problem of LSSVM.

Both methods aim to reduce computational complexity while maintaining competitive predictive performance.

---

## Algorithms Implemented

### TCSMO-LSSVM

Three-Term Sequential Minimal Optimization applied to the dual formulation of LSSVM.

Key characteristics:

- Iterative optimization in the dual space
- Update of three Lagrange multipliers per iteration
- Support vector pruning strategy
- Sparse model generation
- Efficient convergence for large datasets

---

### SCG-LSSVM

Spectral Conjugate Gradient method applied to the dual LSSVM problem.

Key characteristics:

- Iterative solution of the KKT system
- Spectral step-size adaptation
- Reduced memory requirements
- Fast convergence for large-scale problems

---

## Repository Structure
.
├── data/ # Example datasets
├── experiments/ # Experimental scripts
├── models/ # LSSVM implementations
│ ├── tcsmolssvm.py
│ ├── scglssvm.py
├── utils/ # Helper functions
├── notebooks/ # Demonstrations
└── README.md

---

## Installation

Clone the repository: git clone https://github.com/yourusername/lssvm-dual-approach.git


Typical dependencies include:

- numpy
- scipy
- scikit-learn
- matplotlib

---

## Example Usage

Example of training a model using **TCSMO-LSSVM**.

```python
from models.tcsmolssvm import TCSMOLSSVM

model = TCSMOLSSVM(
    kernel='rbf',
    gamma=10,
    sigma=1.0
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from models.scglssvm import SCGLSSVM

model = SCGLSSVM(
    kernel='rbf',
    gamma=10,
    sigma=1.0
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

# Experimental Results

The algorithms were evaluated on several benchmark datasets including:

* UCI Machine Learning Repository datasets

* Synthetic classification datasets

* Large-scale datasets such as ADULT and SHUTTLE

Evaluation metrics include:

* Accuracy

* RMSE

* R²

* Training time

* Sparsity level

Results indicate that the proposed methods achieve competitive predictive performance while significantly reducing training time.
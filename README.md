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

  - **FSLM-LSSVM**  
  A Fixed Sized Levenberg-Marquardt method for iterative pruning of support vectors.

- **RFSLM-LSSVM**  
  A Reduced Fixed Sized Levenberg-Marquardt method operating on a reduced support vector set.

- **LSSVM-ADMM**  
  An LSSVM with L1 (LASSO) regularization in the primal, solved via the Alternating Directions Multipliers Method.

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

### FSLM-LSSVM

Fixed Sized Levenberg-Marquardt method for iterative pruning of LSSVM support vectors, working with the full coefficient matrix at every iteration.

Key characteristics:

- Iterative Levenberg-Marquardt updates in the primal space
- Gradual pruning of support vectors via a sliding window strategy
- Configurable pruning percentage and window size
- Sparse model generation with bias term

---

### RFSLM-LSSVM

Reduced Fixed Sized Levenberg-Marquardt method, extending FSLM-LSSVM by operating on a dynamically updated reduced support vector set.

Key characteristics:

- Works with a reduced subset of support vectors at each iteration
- Periodic replacement of pruned support vectors with new candidates
- Reduced memory and computational cost per iteration
- Configurable reduction rate and window size

---

### LSSVM-ADMM

LSSVM with L1 (LASSO) regularization in the primal formulation, solved via the Alternating Directions Multipliers Method (ADMM).

Key characteristics:

- L1 regularization for automatic sparsity induction
- ADMM-based optimization of the primal problem
- Incomplete Cholesky factorization for kernel approximation
- No bias term in the final solution

---


## Repository Structure
```
.
├── data/ # Example datasets
├── experiments/ # Experimental scripts
├── models/ # LSSVM implementations
│ ├── dual/
│ │ ├── SCG_LSSVM/
│ │ │ └── classifier.py
│ │ └── TCSMO_LSSVM/
│ │ └── classifier.py
│ ├── primal/
│ │ ├── FSLM_LSSVM/
│ │ │ └── classifier.py
│ │ ├── RFSLM_LSSVM/
│ │ │ └── classifier.py
│ │ └── LSSVM_ADMM/
│ │ └── classifier.py
│ └── base.py
├── utils/ # Helper functions
├── notebooks/ # Demonstrations
└── README.md
```

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
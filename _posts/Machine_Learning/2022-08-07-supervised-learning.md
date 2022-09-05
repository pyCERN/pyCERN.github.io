---
title: Supervised Learning
use_math: true
date: 2022-08-07
categories:
  - Machine Learning
---


# Supervised Learning
_Supervised Learning_: From the training data $(X, Y)$, make a good prediction of the output $Y$, $\hat{Y}$


## Simple Approaches for Supervised Learning
- Parametric - Least squares
- Non-parametric - Nearest neighbors


### Least Squares
Assumption of the linear model:
  - Input: $X = (1, x_1, x_2, ..., x_d)^T$
  - Output: $\hat{Y} = X^T w = w_0 + \sum\limits_{j=1}^d x_j^T w_j$
    - $w$: Parameter of the model
    - $w_0$: Bias

Fitting the model to data $\to$ _least square_ method
  - Goal: Pick $w$ to minimize the _residual sum of squares_ (RSS)

$$
\begin{aligned}
    RSS(w) &= (\vec{y} - Xw)^T (\vec{y} - Xw) \\
    \frac{\partial}{\partial w} RSS(w) &= X^T (\vec{y} = Xw) = 0 \\
    \therefore w &= (X^T X)^{-1} X^T \vec{y} \quad \text{if $X^T X$ is non-singular}
\end{aligned}
$$


### Nearest Neighbor Methods
Direct output from input $x_i$ based on _closeness_
  - $\hat{Y} = \frac{1}{k} \sum\limits_{x_i \in N_k(x)} y_i$
    - $N_k(x)$: $k$-closest points to $x$, neighborhood of $x$

| Least squares | Nearest neighbors |
|---|---|
| - Strong assumption (linear decision boundary) on data <br> - Low variance (stable), high bias (inaccurate) | - Can adapt to any situation <br> - High variance, low bias

Most ML methods are based on these two techniques.
  - _Kernel methods_: Capture features of high dimension data
  - Non-linear basis $\phi(\vec{x})$ rather than just $x$ for linear models


## Statistical Decision Theory
_Loss function_ $L(Y, f(X))$: Given the model $f$, penalizes errors in prediction $f(X)$
  - _Expected error loss_ (EPE): $L(Y, f(X)) = (Y - f(X))^2$

Goal: Choose $f$ that minimizes the loss function

For EPE loss function, pointwise expectation $f(x) = E(Y|X = x)$. $\to$ _regression function_

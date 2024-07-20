## Matrix inverse
In order to determine inverse matrix, follow these steps:
In this exercise, a 2x2 matrix is provided for simply explaination

**Given matrix A. Step 1: Find determinant of A.**

Given Matrix **A**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{A}=\begin{bmatrix}a&b\\c&d\end{bmatrix},\quad\mathbf{A}\in\mathbb{R}^{2\times2}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?det(\mathbf{A})=ad-bc" />
</p>

**Step 2: Depending on determinant of A, we will appoarch appropriately**

If det(A) is not equal to 0, it is invertible. Otherwise, conclude that there is no inverse matrix for given matrix.

**Step 3: In case of det(A) which is not 0, determine inverse matrix by following equation:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{A}^{-1}=\frac{1}{\det(\mathbf{A})}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}" />
</p>

## Eigenvector and eigenvalues
In math, eigenvector is a vector which remains unchanged stage after a linear transformation. The result is a scaled
version of original vector.
Application: Dimensionality Reduction, Clustering, ...
**Guidelines: Given nxn matrix and one identity matrix (has the indentical shape with that matrix).**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{A}\in\mathbb{R}^{n\times n},\ \mathbf{I}\ (\text{identity\ matrix})\in\mathbb{R}^{n\times n},\ \mathbf{v}\in\mathbb{R}^{n}" />
</p>

**Step 1: Determine eigenvalues**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\text{Eigenvalue}\ (\lambda):\ \det(\mathbf{A}-\lambda\mathbf{I})=0" />
</p>

**Step 2: Determine eigenvector (v)**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\text{Eigenvector}\ (\mathbf{v}):\ \mathbf{A}\mathbf{v}=\lambda\mathbf{v}\iff(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=0" />
</p>

**Step 3: Normalize vector**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\frac{\mathbf{v}}{\|\mathbf{v}\|},\ v_i=\frac{v_i}{\sqrt{\sum_{i=1}^n{v_i^2}}}" />
</p>

Note: numpy provide function called **np.linalg.eig(A)** that help us find eigenvector and eigenvalue quick.

**Based on given equations, we are gonna try with simple example**
Given matrix 
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}-6&3\\4&5\end{bmatrix}" />
</p>
and identical matrix:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1&0\\0&1\end{bmatrix}" />
</p>

**Step 1: solve eigen values:**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\left|\begin{bmatrix}-6&3\\4&5\end{bmatrix}-\lambda\begin{bmatrix}1&0\\0&1\end{bmatrix}\right|=0" />
</p>

**We get**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\lambda=-7\ \text{or}\ 6" />
</p>

**Step 2: With eigen value is 6, find eigen vector**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}-6&3\\4&5\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=6\begin{bmatrix}x\\y\end{bmatrix}" />
</p>

**Solve x, y accordingly, we get**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}1\\4\end{bmatrix}" />
</p>

## Cosine similarity**
Cosine similarity is a measure of similarity between two non-zero vectors
Application: Clustering, Information retrieval
It is actually simple to find this value based on given guide line
**Expression formular:**
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?cs(\mathbf{x},\mathbf{y})=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}=\frac{\sum_{i=1}^n{x_i y_i}}{\sqrt{\sum_{i=1}^n{x_i^2}}\sqrt{\sum_{i=1}^n{y_i^2}}}" />
</p>

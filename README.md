# Linear Regression From Scratch (Gradient Descent)

## Project Overview
This project implements **Simple Linear Regression from scratch using Python**, without using machine learning libraries such as `scikit-learn`.  
The objective is to understand the **mathematical foundations and optimization process** behind linear regression.

The model predicts **Salary based on Years of Experience** using **gradient descent**.

---

## Dataset
- **Source:** Kaggle  
- **File:** `Salary_Data.csv`
- **Features:**
  - `YearsExperience` (independent variable)
  - `Salary` (dependent variable)
- **Number of observations:** 30

---

## Mathematical Model

### Hypothesis Function
\[
f(x) = wx + b
\]

Where:
- \( w \) is the weight (slope)
- \( b \) is the bias (intercept)

---

### Cost Function (Mean Squared Error)
\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x_i) - y_i)^2
\]

---

### Gradients
\[
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)x_i
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)
\]

---

### Gradient Descent Update Rules
\[
w := w - \alpha \frac{\partial J}{\partial w}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]

Where \( \alpha \) is the learning rate.

---

## Implementation Details
- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib
- **Machine Learning Libraries Used:** None

All components including the cost function, gradients, and optimization loop are implemented manually.

---

## Training Configuration
- **Learning Rate:** 0.01
- **Iterations:** 10,000
- **Initial Parameters:**  
  \( w = 0 \), \( b = 0 \)

### Final Learned Parameters
w = 9449.9623
b = 25792.2002


---

## Visualization
- Scatter plot of the training data
- Regression line learned using gradient descent

This confirms correct convergence of the model.

---

## Project Structure
Project_1/
├── linearRegression.ipynb
├── Salary_Data.csv
├── README.md
└── .gitignore

yaml
Copy code

---

## Key Learning Outcomes
- Implemented linear regression from first principles
- Understood gradient descent optimization
- Gained hands-on experience with mathematical ML foundations
- Visualized model behavior and convergence

---

## Future Improvements
- Plot cost vs iterations
- Feature normalization
- Vectorized NumPy implementation
- Extend to multivariate linear regression
- Compare with `scikit-learn` implementation

---

## Author
**Harkamal Toor**  
Aspiring Data Scientist

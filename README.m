# CodeAlpha - Disease Prediction from Medical Data

Breast cancer prediction (malignant vs benign) using classical machine learning models.
Task 4 of the CodeAlpha Machine Learning Internship.

## Dataset
Breast Cancer Wisconsin (Diagnostic): 569 samples, 30 numeric features.
Loaded from `sklearn.datasets.load_breast_cancer`.

## Workflow
1. Exploratory data analysis
2. Stratified train/test split (80/20) and feature scaling
3. Models: Logistic Regression, Random Forest, SVM, XGBoost
4. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| Random Forest | 0.956 | 0.959 | 0.972 | 0.966 | 0.993 |
| SVM | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| XGBoost | 0.956 | 0.947 | 0.986 | 0.966 | 0.993 |

## Key findings
- All models reach ROC-AUC of about 0.99.
- Most important features: worst perimeter, worst area, worst concave points.
- A false negative (malignant tumor predicted as benign) is the costliest error, so recall on the malignant class was examined alongside accuracy.

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Notebook
[CodeAlpha_DiseasePrediction.ipynb](CodeAlpha_DiseasePrediction.ipynb)

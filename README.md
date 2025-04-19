# Loan Approval Classification using Machine Learning

![ChatGPT Image Apr 19, 2025, 10_20_28 AM](https://github.com/user-attachments/assets/88f2cbde-e637-4de6-9a55-9433b621ccff)

---

## Overview

This project aims to streamline the **loan approval process** using supervised **Machine Learning algorithms**.  
We developed and compared multiple models to classify loan applications into *Approved* or *Denied* categories, utilizing real-world financial applicant data.

By leveraging feature engineering, model selection techniques, and evaluation metrics, we identified the best-performing model with optimized predictive power.

---

## Key Features

- **Data Preprocessing**: Handling categorical variables (Ordinal & One-hot Encoding) and feature scaling.
- **Feature Selection**: 
  - **Backward Elimination** and **Forward Selection** based on **AIC score**.
  - Final reduced set: **12 high-impact predictors**.
- **Models Implemented**:
  - Logistic Regression
  - Decision Tree (CART)
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (KNN)
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion Matrices, Gains & Lift Charts, ROC Curves
- **Best Model**: 
  - **K-Nearest Neighbors** with **k=16**, achieving **90%+ accuracy**.

---

## Project Structure

```bash
Loan-Approval-Classification-ML-Project/
├── ML_Final_Project_Report.pdf         # Detailed Research Report
├── ML_Project_PPT.pdf                   # Final Project Presentation
├── Loan_Approval_Classification.ipynb   # Google Colab Notebook
├── README.md                            # Project Documentation
├── LICENSE                              # MIT License
└── requirements.txt                     # Python Dependencies
```

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:-----|:--------:|:---------:|:------:|:--------:|:-------:|
| KNN (k=16) | 90.4% | 86.1% | 68.0% | **76.0%** | 82.4% |
| CART | 89.7% | 82.2% | 68.9% | 74.9% | 82.3% |
| Logistic Regression | 88.7% | 76.4% | 71.8% | 74.0% | 82.7% |
| Naive Bayes | 74.9% | 47.0% | 97.9% | 63.5% | 83.1% |

**KNN model outperformed other classifiers**, offering a balanced trade-off between precision and recall.

---

## Skills Demonstrated

- Machine Learning (Classification)
- Feature Engineering & Dimensionality Reduction
- Model Evaluation and Selection
- Data Visualization (Matplotlib, Seaborn)
- Python Programming (pandas, numpy, scikit-learn, dmba)
  
---

## License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, share, and modify it for academic and professional purposes.

---

## Contact

Feel free to connect on [LinkedIn](#) for any queries or collaborations.

---

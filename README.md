# Predictive-Modeling-for-E-Commerce-Customer-Churn
Aim: proactive customer retention

Brief Overview: Customer churn prediction project using ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost). Includes data preprocessing, feature engineering, hyperparameter tuning. XGBoost achieved high Recall, PR AUC, and F1 Score. SHAP analysis explains key churn drivers. 
******************************************************************************************************************************************************************************************************************************************************************************************************
This project demonstrates the process of predicting customer churn using various machine learning techniques. Customer churn, or attrition, is a critical metric for businesses as retaining existing customers is often more cost-effective than acquiring new ones. By building a predictive model, businesses can proactively identify customers at risk of churning and implement targeted retention strategies.

The project follows a standard machine learning pipeline:
1. Data Loading and Exploration
2. Data Preprocessing and Feature Engineering
3. Model Building and Evaluation (Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machines)
4. Hyperparameter Tuning
5. Model Comparison and Selection
6. Model Interpretation (SHAP Analysis)

The goal is to build a model that effectively identifies customers likely to churn, with a focus on metrics like Recall, Precision-Recall AUC, and F1 Score, especially important for imbalanced datasets.

METRICS

In the context of our model, correctly identifying customers likely to churn (true positives) is particularly important because it allows the business to take timely action. False negatives (failing to identify customers who will churn) represent missed opportunities to retain valuable customers and the associated revenue.

While it's also important not to misclassify loyal customers as potential churners (false positives), the cost of reaching out to a satisfied customer is generally lower than the cost of losing a customer who could have been retained with proper intervention.
This is why I will use metrics that emphasize the correct identification of churners instead of accuracy.
To create model, that correctly predicting the churned customers I am going to focusing on these metrics:

Recall: This is crucial as it directly measures how well we're identifying churned customers.

PR AUC: This gives a good overall picture of performance on imbalanced data.

F1 Score: This balances precision and recall

DATASET

The dataset used in this project is from Kaggle: [E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction). It contains various customer attributes and a target variable indicating whether a customer has churned.

METHADOLOGY

The project explores several classification algorithms to predict churn:

Logistic Regression: Used as a baseline model.

Decision Tree Classifier: Explored for its interpretability.

Random Forest Classifier: An ensemble method to improve upon Decision Trees.

XGBoost Classifier: A powerful gradient boosting algorithm known for high performance.

Support Vector Machines (SVM): A versatile algorithm for various classification tasks.

This project enables hyperparamter tuning for models like Logistics Regression, Decision Trees, Random Forest, SVM using GridSearchCV and XGBoost, using optuna.

To handle the imbalanced nature of the dataset, techniques like class weighting and undersampling (TomekLinks) were incorporated into the modeling pipeline. Feature engineering was also performed to create new variables that potentially improve model performance. Hyperparameter tuning using Randomized Search was applied to optimize the models.


RESULTS

Several models were evaluated based on their performance on a validation set. The XGBoost Classifier consistently showed the best performance across key metrics, including:

Accuracy: 98.10% 

Recall (ability to identify actual churners): 0.95

Precision-Recall AUC (performance on imbalanced data):0.9919

F1 Score (harmonic mean of precision and recall): 0.95

The final evaluation of the best model (XGBoost) on the held-out test set yielded impressive results, demonstrating its strong capability in predicting customer churn.

 MODEL INTERPRETATION
 
To understand how the best model makes predictions, SHAP (SHapley Additive exPlanations) analysis was performed. This involved:

Calculating SHAP values for the test set.

Generating a SHAP summary plot to visualize overall feature importance.

Generating SHAP dependence plots to understand the relationship between specific features and the model's output.

The SHAP analysis provides valuable insights into which features are most influential in predicting churn and how their values impact the prediction.

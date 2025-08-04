# PGA Tour Performance Analysis

## Overview
This project applies machine learning and data analytics to model PGA Tour player performance, leveraging historical Strokes Gained metrics and course features. The goal is to predict the likelihood of a player finishing in the Top 5, Top 10, or Top 20 of a specific tournament, helping identify key performance drivers for success.

## Full Data Science Workflow:
  - Data Acquisition & CLeaning: Merged multiple PGA datasets, handled missing values, and normalized features.
  - Feature Engineering & Selection: Created predictive variables from Strokes Gained stats and course difficulty indicators. Then, I performed correlation analysis and calculated Variance Inflation Factor (VIF) to detect multicollinearity and refine feature set.
  - Modeling & Evalutation: Tested and compared multiple algorithms (Logistic Regression, Random Forest, and XGBoost), using ROC-AUC as the primary metric. Final selection was XGBoost, achieving ROC-AUC up to 0.85 for Top 10 finish predictions.
  - Visualization & Insights: Built an interactive Tableau dashboard for course difficulty analysis and player performance trends.

## Tech Stack
Languages: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-Learn, XGBoost
Tools: Jupyter Notebook (VS Code), Tableau Public

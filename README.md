# PGA Tour Performance Analysis

## Overview
This project applies machine learning and data analytics to model PGA Tour player performance, leveraging historical Strokes Gained metrics and course features. The goal is to predict the likelihood of a player finishing in the Top 5, Top 10, or Top 20 of a specific tournament, helping identify key performance drivers for success.

## Full Data Science Workflow:
  - Data Acquisition & CLeaning: Merged multiple PGA datasets, handled missing values, and normalized features.
  - Feature Engineering & Selection: Created predictive variables from Strokes Gained stats and course difficulty indicators. Then, I performed correlation analysis and calculated Variance Inflation Factor (VIF) to detect multicollinearity and refine feature set.
  - Modeling & Evalutation: Tested and compared multiple algorithms (Logistic Regression, Random Forest, and XGBoost), using ROC-AUC as the primary metric. Final selection was XGBoost, achieving ROC-AUC up to 0.85 for Top 10 finish predictions.
  - Visualization & Insights: Built an interactive Tableau dashboard for course difficulty analysis and player performance trends.

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-Learn, XGBoost
- **Tools:** Jupyter Notebook (VS Code), Tableau Public

## Validation Strategy: Stratified K-Fold vs TimeSeriesSplit
At first, I used Stratified K-Fold cross-validation because it’s common for classification problems and ensures balanced class distribution in each fold. It gave pretty solid ROC-AUC scores (around 0.80+), which looked great on paper.

But then I realized this isn’t realistic for a forecasting problem like predicting PGA results. Stratified K-Fold randomly shuffles data, which means the model can "peek into the future" during training. That’s a big issue if you want to simulate real-world performance.

So, I switched to **TimeSeriesSplit**, which respects chronological order. It trains on past events and tests on future events — exactly how this model would be used in real life. The trade-off? Scores dropped (ROC-AUC around 0.50–0.55), but that’s the honest truth: predicting golf outcomes with limited features and real-world constraints is tough.

**Why keep the lower scores?** Because it shows I understand the importance of time-based validation and I’m not just chasing pretty numbers. 

## Future Improvements
- **Live Data Pipeline:** Connect to the DataGolf API for automatic updates and real-time predictions. 
- **Weather:** Use DataGolf API to pull real-time weather conditions
- **Recent Form Features:** Add rolling averages for strokes gained metrics (e.g., last 3 and last 5 tournaments) to capture momentum.
- **Hyperparameter Tuning & Imbalance Handling:** Optimize XGBoost parameters and use class weights to handle the natural imbalance in Top 5 / Top 10 outcomes.
- **Model Comparisons:** Experiment with LightGBM or CatBoost and maybe even a neural network for fun

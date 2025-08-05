# PGA Tour Performance Analysis

## Overview
This project applies machine learning and data analytics to model PGA Tour player performance using historical Strokes Gained metrics and course features. The goal is to predict the likelihood of a player finishing in the Top 5, Top 10, or Top 20 of a specific tournament. Along the way, the project explores key performance drivers, model evaluation strategies, and realistic forecasting challenges.

## Full Data Science Workflow:
  - **Data Acquisition & Cleaning:** Merged multiple PGA datasets, handled missing values, and normalized features.
  - **Feature Engineering & Selection:** Created predictive variables from Strokes Gained stats and course difficulty indicators. Then, I performed correlation analysis and calculated Variance Inflation Factor (VIF) to detect multicollinearity and refine feature set.
  - **Modeling & Evalutation:** Tested and compared multiple algorithms (Logistic Regression, Random Forest, and XGBoost), using ROC-AUC as the primary metric. Final selection was XGBoost for its performance and ability to handle feature interactions.
  - **Validation:** Compared Stratified K-Fold (traditional) vs TimeSeriesSplit (realistic for forecasting).
  - **Visualization & Insights:** Built an interactive Tableau dashboard for course difficulty analysis and player performance trends.

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-Learn, XGBoost
- **Tools:** Jupyter Notebook (VS Code), Tableau Public

## Validation Strategy: Stratified K-Fold vs TimeSeriesSplit
At first, I used Stratified K-Fold cross-validation because it’s common for classification problems and ensures balanced class distribution in each fold. It gave pretty solid ROC-AUC scores (around 0.80+), which looked great on paper.

But then I realized this isn’t realistic for a forecasting problem like predicting PGA results. Stratified K-Fold randomly shuffles data, which means the model can "peek into the future" during training. That’s a big issue if you want to simulate real-world performance.

So, I switched to **TimeSeriesSplit**, which respects chronological order. It trains on past events and tests on future events — exactly how this model would be used in real life. While TimeSeriesSplit was used to simulate realistic evaluation, the unusually high ROC-AUC scores (0.80–0.90) may indicate residual data leakage or incomplete chronological sorting. Future improvements include stricter date ordering and exclusion of post-event data to ensure true predictive performance.


## Future Improvements
- **Live Data Pipeline:** Connect to the DataGolf API for automatic updates and real-time predictions. 
- **Weather & Course Conditions:** Use DataGolf API to pull real-time weather conditios and precise course metrics
- **Recent Form Features:** Add rolling averages for strokes gained metrics (e.g., last 3 and last 5 tournaments) to capture momentum.
- **Hyperparameter Tuning & Imbalance Handling:** Optimize XGBoost parameters and use class weights to handle the natural imbalance in Top 5 / Top 10 outcomes.
- **Model Comparisons:** Experiment with LightGBM or CatBoost and maybe even a neural network for fun

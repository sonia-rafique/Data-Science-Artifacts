
### Bank Term Deposit Prediction (Marketing Optimization)
## Project Objective
The goal is to predict whether a customer will subscribe to a term deposit based on bank marketing campaign data. By understanding customer behavior, the bank can target the right audience, reduce marketing costs, and improve conversion rates.

## My Approach
I followed a structured data science workflow to build a reliable and interpretable classification model:

1. **Data Exploration & Transformation:** Checked the data structure and handled high-cardinality categorical features (like job and education) using advanced target encoding. Time-based fields were mapped cyclically to keep their natural sequence.
2. **Class Imbalance Management:** Since the majority of customers did not subscribe, I used a stratified split and mathematically adjusted the model's cost function (`scale_pos_weight`) to make sure the minority class wasn't ignored.
3. **Model Training:** Utilized a highly optimized **LightGBM Classifier** with early stopping to avoid overfitting.
4. **Explainable AI (XAI):** Implemented **SHAP** to audit individual customer profiles and understand why the model made specific predictions.

## Key Findings & Results
### 1. Model Performance
The model achieved an overall accuracy of **89%**. While it is exceptionally good at identifying customers who won't subscribe (**99% recall**), it filters out the high-potential subscribers with a precision of **55%**.
### 2. What Drives Customer Decisions? 
* **Call Duration is King:** The longer a customer stays on the phone (`duration`), the more likely they are to subscribe. This is the strongest predictor.
* **Past Behavior Matters:** Customers who responded positively to previous marketing campaigns (`poutcome`) showed a significantly higher intent to subscribe again.
* **Timing & Context:** The month of contact and the customer’s economic background (balance/job) play a major role in their decision-making.

-----------------------------------------------------------------------------------------------------------------------------------
# Mall Customer Segmentation Analysis

## Task Objective
The primary objective of this project is to analyze raw retail customer data to discover distinct consumer segments based on demographic features and spending behaviors. By identifying these underlying groups through unsupervised machine learning, a business can transition from broad marketing approaches to highly targeted, data-driven CRM strategies that improve customer retention and maximize lifetime value.

## MY Approach
The analytical workflow was structured into five key phases to ensure an empirical and organized data science process:

1. **Exploratory Data Analysis and Cleaning**
   The corporate data was initially audited for architectural structural integrity. Column headers were stripped of structural white spaces, and important consumer metrics (Annual Income and Spending Score) were converted to clean numeric data types. Row segments containing invalid null vectors were filtered out to ensure statistical validity. Bivariate scatter plots and density boxplots were rendered to understand the internal relationship between variables.

2. **Feature Scaling and Optimization Diagnostics**
   To eliminate mathematical bias caused by differing unit scales between age, annual income, and spending parameters, the empirical metrics were normalized utilizing standard scaling. To establish the mathematical framework for cluster boundaries, both the Elbow Method (Within-Cluster Sum of Squares) and the Geometric Silhouette Score metrics were calculated iteratively across an evaluated cluster range of K values from 2 to 10. 

3. **K-Means Model Execution**
   By mapping the structural inflection point of the elbow line against the highest stable silhouette score, an optimal structural cluster count of K=5 was established. A production-grade K-Means clustering algorithm was executed with k-means++ initialization parameters over 10 random seeds to establish robust spatial cluster centroids.

4. **Dimensionality Reduction and Visual Mapping**
   Because the feature space consists of three dimensions (Age, Income, Spending Score), Principal Component Analysis (PCA) was used for dimensionality reduction. The high-dimensional cluster results and their absolute spatial centroids were projected onto a 2D coordinate system for visual interpretation.

5. **Customer Persona Mapping and Strategy Formulation**
   A cluster summary matrix was compiled by computing the quantitative means of each identified group. These data-driven definitions were mapped to specific consumer personas, and unique operational marketing strategies were developed for each segment. Finally, individual customer cohorts were exported into clean data tables for targeted CRM execution.

## Results and Findings
### 1. Empirical Cluster Profile Summary
The empirical evaluation partitioned the mall consumer database into five distinct cohorts with the following localized feature averages:

* **Cluster 0: Budget Frugals**
  An older cohort with low average annual income and a low spending index.
* **Cluster 1: Resourceful Trendsetters**
  A younger consumer demographic characterized by moderate annual earnings paired with highly elevated spending scores.
* **Cluster 2: Premium High-Rollers**
  A middle-aged premium demographic possessing both substantial annual income and an aggressive spending score.
* **Cluster 3: Conservative Affluents**
  A mature, affluent consumer demographic with elevated annual earnings but highly restrained spending patterns.
* **Cluster 4: Stable Traditionalists**
  A mature, well-balanced group showing moderate annual earnings matched by moderate, stable spending scores.

### 2. Strategic Marketing Playbook
Based on the distinct spending behaviors identified above, the following tactical business interventions are recommended:
* **Budget Frugals (Cluster 0)**
  Target with essential value bundles, clearance sale alerts, and low-tier discount structures to encourage consistent basic spending.
* **Resourceful Trendsetters (Cluster 1)**
  Engage heavily via interactive digital social campaigns, flash product sales, and flexible Buy Now Pay Later (BNPL) financial integrations.
* **Premium High-Rollers (Cluster 2)**
  Enroll into elite concierge tiers, provide early access to exclusive luxury products, and assign premium account managers to sustain lifetime loyalty.
* **Conservative Affluents (Cluster 3)**
  Promote value-driven rewards such as cashback credit card partnerships, multi-device ecosystem warranties, and long-term asset protections.
* **Stable Traditionalists (Cluster 4)**
  Engage with structured milestone loyalty point trackers, direct physical mail updates, and ironclad customer service guarantees.


----------------------------------------------------------------------------------------------------------------------------------

# Task 3 ; Household Energy Consumption Time Series Forecasting

## Task Objective
The primary objective of this project is to build an analytical framework capable of forecasting short-term household energy utilization based on sequential historical records. By effectively predicting variations in global active power demand, utility infrastructure and property managers can make informed adjustments to power distribution, manage heavy electrical loads efficiently, and optimize grid resource allocation.

## My Approach
The development of this forecasting engine was executed through a structured, reproducible four-stage machine learning process:

1. **Data Ingestion and Engineering Cleanup**
   The raw dataset was audited for structural inconsistencies, and timestamps were consolidated into a unified index. The operational timeline was standardized using Pandas to handle invalid null arrays. The granularity of the data was downsampled from localized minute-by-minute entries to an hourly frequency to stabilize sudden operational noise while maintaining structural intra-day variations.

2. **Temporal Feature Construction**
   To assist machine learning structures in capturing repeating time patterns, the absolute datetime index was decomposed into distinct independent variables. These engineered features included continuous indicators for the hour of the day, day of the week, calendar month, and a binary classifier distinguishing standard weekdays from weekend profiles.

3. **Multi-Model Implementation Strategy**
   The structured timeline was separated chronologically into training vectors and a distinct validation testing period to ensure proper validation without data leakage. Three independent, diverse modeling methodologies were developed to compare forecasting capabilities:
   * **Statistical Baseline (Holt-Winters Exponential Smoothing):** Implemented to model exponential trends and baseline season structures.
   * **Classical Statistical Vector (ARIMA):** Configured via autocorrelation parameters to evaluate linear temporal dependencies.
   * **Machine Learning Regressor (XGBoost / Gradient Boosting):** Deployed to capture complex, non-linear relationships across the newly engineered time features.

4. **Quantitative Evaluation Framework**
   Model predictions were aligned alongside the real holdout test records. Performance was strictly validated using standardized regression metrics, specifically Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), accompanied by structural line graphs overlaying forecasted predictions against actual usage.

## Results and Findings

### 1. Quantitative Performance Benchmarks
The evaluated forecasting structures demonstrated varying degrees of success in capturing the hourly shifts in household energy usage. The performance metrics across the test window are detailed below:

* **Gradient Boosting / XGBoost Framework**
  * Mean Absolute Error (MAE): 0.38 kW
  * Root Mean Squared Error (RMSE): 0.51 kW
  * Performance Context: This model outperformed the other approaches, achieving the lowest overall error. By leveraging the explicit hourly and weekday features, it closely tracked peak utilization hours and structural weekend behavior changes.

* **Holt-Winters Exponential Smoothing**
  * Mean Absolute Error (MAE): 0.44 kW
  * Root Mean Squared Error (RMSE): 0.58 kW
  * Performance Context: Exhibited stable baseline performance. It successfully adapted to smooth repeating daily cycles but failed to react dynamically to sudden, irregular shifts in consumer power demand.

* **ARIMA (Statistical Framework)**
  * Mean Absolute Error (MAE): 0.50 kW
  * Root Mean Squared Error (RMSE): 0.65 kW
  * Performance Context: Displayed the highest error metrics in this scenario. While competent at capturing localized linear correlations, it struggled to scale across rapid, non-linear household consumption fluctuations over an extended prediction horizon.

### 2. Analytical Insights and Key Conclusions
* **Feature Value:** The high predictive accuracy of the Gradient Boosting approach demonstrates that short-term household energy consumption is highly dependent on human routines (such as morning preparations or evening returns), which are best captured by explicit hour-of-day and weekday features.
* **Error Profile:** All models exhibited higher residual errors during unexpected, sharp drops or spikes in usage, indicating that incorporating external variables—such as localized temperature readings or holiday schedules—could improve future model robustness.

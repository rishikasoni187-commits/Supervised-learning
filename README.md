# Supervised-learning



1.**Project Title** - *“Predictive Modeling Using Supervised Machine Learning Techniques for Classification and Decision Support”*



**2. Description of Data**


*   Data Source - https://www.kaggle.com/datasets/satyaworks/perishable-goods-supply-chain-and-transport-dataset
*   Data Size: **26.1 MB**
*  Data Type: **Cross-sectional**
*  Data Dimension:
     * Number of Variables - **32**
     * Number of Observations - **100,000**

*   Data Variable Type:

    **Numeric**-
    *  Integer - **11 columns**
    
    (Daily Production/Quantity, No. of Vehicles Used Per Day, Trip Frequency, Average Distance Travelled (km), Temperature Maintained (°C), Humidity Maintained (%), Daily Demand, Shelf Life (Days), Transportation Cost per Trip, Storage Cost, Wastage Cost). **All are 100,000 non-null.**
    *  Float - **5 columns**

      (Vehicle Capacity (kg/ton), Transport Duration (hours), Unloading Time, Total Supply Chain Time, Spoilage Percentage (%)). **All are 100,000 non-null.**

     **Non-Numeric** -
       *   Object -
    **16 columns**

    (Product Name, Category, Transportation Method, Vehicle Type, Packaging Method, Time of Harvest, Time to Loading, Source Location, Destination Location, Who Receives the Goods?, End User Type, Is Real-time Tracking Used?, Tracking Method, Data Collected, Initial Quality Grade, Reasons for Spoilage). *Note* - **Tracking Method has 50,125 non-null entries, suggesting missing values.**
     
* Data Variable Category-I:

1.   **Non-Categorical**: All Numeric Values (integer + Float)

2.   **Categorical** -  
     *  **Nominal** -  Product Name, Category, Transportation Method, Vehicle Type, Packaging Method, Time of Harvest, Time of Loading, Source Location, Destination Location, Who Receives the Goods?, End user type, Is Real time Tracking Used?, Tracking Method, Data Collected, Reasons For Spoilage
     *  **Ordinal** - Inital Quality Grade
*  Data Variable Category-II:
    For the specified unsupervised clustering task, all columns in the dataset were uniformly classified as **'Input Variable/Feature'**.

*  Brief about Dataset:
    The dataset captures operational and logistical characteristics of perishable goods supply chain and transportation activities. It includes information related to product categories, transportation modes, distances, costs, and delivery performance. The dataset is suitable for clustering analysis to identify distinct operational segments and patterns within supply chain entities, enabling better managerial decision-making.

    *Refer to Code 1*  


 **3. Project Objectives**

*  Classification of Dataset into
      *  Segments
      *  Clusters
      *  Classes

 using Supervised Learning Classification Algorithms

* **Identification of {Important | Contributing | Significant**}
Variables or Features and their Thresholds for Classification

* **Determination of an appropriate Classification Model based on Performance Metrics**

### **For Analysis:**
5001 Random Records were created using last 3 digits of Roll Number as “random_state” everywhere.
roll no. - 065099
dataframe name was created using 1st two letters of name (rishika soni) + last 3 digits of roll no. **rs099**


**4. Analysis of Data**

Refer To the Analytical Report
Link- https://colab.research.google.com/drive/1y5kpK8sKNqmAZ7G9aBmlCUAJVyGmUiOg?usp=sharing

### **Observations and Findings**


## Comprehensive Summary of Supervised Machine Learning Models for Classification

This section provides a consolidated summary of the performance, run statistics, cross-validation results, and feature importance for all implemented classification models (Logistic Regression, Support Vector Machine, Stochastic Gradient Descent Classifier, Decision Tree, K-Nearest Neighbors, Random Forest, and Extreme Gradient Boosting). It also includes a comparative analysis and overall insights regarding the model appropriateness and next steps.

---

### 1. Summarized Model Performance and Run Statistics

The table below consolidates the key performance metrics, including accuracy, F1-score (weighted average), training time, prediction time, and mean cross-validation accuracy for all evaluated models. The mean cross-validation accuracy provides a more robust estimate of the model's generalization performance on unseen data.

| Model                  | Accuracy | F1-Score (Weighted) | Training Time (s) | Prediction Time (s) | Mean CV Accuracy |
|:-----------------------|:---------|:--------------------|:------------------|:--------------------|:-----------------|
| XGBoost Classifier     | 0.2046   | 0.2040              | 1.4667            | 0.0204              | 0.2090           |
| K-Nearest Neighbors    | 0.2046   | 0.1972              | 0.1130            | 0.0000              | 0.2020           |
| Decision Tree          | 0.2166   | 0.2165              | 0.2088            | 0.0000              | 0.2020           |
| Logistic Regression    | 0.1966   | 0.1957              | 0.1084            | 0.0009              | 0.2014           |
| Support Vector Machine | 0.1910   | 0.1899              | 3.0178            | 0.2009              | 0.1998           |
| Random Forest          | 0.2006   | 0.1997              | 1.7513            | 0.0393              | 0.1972           |
| SGD Classifier         | 0.1902   | 0.1870              | 0.2615            | 0.0012              | 0.1964           |

**Key Observations from Performance Table:**
*   **Overall Low Performance:** All models exhibit very low accuracy and F1-scores, generally ranging between 19% and 21%. This indicates that the classification task for 'Initial Quality Grade' is highly challenging with the current data and default/basic model configurations.
*   **Top Performers (CV Accuracy):** XGBoost Classifier showed the highest mean cross-validation accuracy (0.2090), followed closely by K-Nearest Neighbors and Decision Tree (both approximately 0.2020).
*   **Efficiency:** Logistic Regression and KNN were among the fastest in terms of training time, while SVM had the longest training time. SGD Classifier had the fastest prediction time.
*   **Consistency:** The mean CV accuracy values are generally very close to the test set accuracies, suggesting that the initial train-test split performance is largely representative.

---

### 2. Identified Important Features and Thresholds (from Decision Tree)

The Decision Tree model, with a `max_depth=4`, provided insights into the most important features and their decision rules for predicting 'Initial Quality Grade':

**Features with High Importance (Most Relevant):**

| Feature                      | Importance |
|:-----------------------------|:-----------|
| Vehicle Type                 | 0.1710     |
| Transport Duration (hours)   | 0.1589     |
| Transportation Cost per Trip | 0.1371     |
| Humidity Maintained (%)      | 0.1274     |
| Daily Demand                 | 0.1234     |
| Wastage Cost                 | 0.1152     |
| Total Supply Chain Time      | 0.1038     |
| Storage Cost                 | 0.0633     |

These features collectively explain a significant portion of the variability in the 'Initial Quality Grade' as captured by this specific Decision Tree model.

**Decision Rules and Thresholds:**
The Decision Tree model's rules provide interpretable thresholds for classification:
```
|--- Transportation Cost per Trip <= 1880.50
|   |--- Total Supply Chain Time <= 6.90
|   |   |--- class: 3
|   |--- Total Supply Chain Time >  6.90
|   |   |--- Vehicle Type <= 0.50
|   |   |   |--- class: 0
|   |   |--- Vehicle Type >  0.50
|   |   |   |--- class: 2
|--- Transportation Cost per Trip >  1880.50
|   |--- Daily Demand <= 874.00
|   |   |--- Vehicle Type <= 1.50
|   |   |   |--- Total Supply Chain Time <= 16.85
|   |   |   |   |--- class: 3
|   |   |   |--- Total Supply Chain Time >  16.85
|   |   |   |   |--- class: 1
|   |   |--- Vehicle Type >  1.50
|   |   |   |--- Storage Cost <= 2225.00
|   |   |   |   |--- class: 4
|   |   |   |--- Storage Cost >  2225.00
|   |   |   |   |--- class: 2
|   |--- Daily Demand >  874.00
|   |   |--- Wastage Cost <= 1666.50
|   |   |   |--- Transport Duration (hours) <= 17.75
|   |   |   |   |--- class: 4
|   |   |   |--- Transport Duration (hours) >  17.75
|   |   |   |   |--- class: 4
|   |   |--- Wastage Cost >  1666.50
|   |   |   |--- Humidity Maintained (%) <= 93.50
|   |   |   |   |--- class: 0
|   |   |   |--- Humidity Maintained (%) >  93.50
|   |   |   |   |--- class: 2
```
**Features with Very Low or Zero Importance:** A large number of features (e.g., 'No. of Vehicles Used Per Day', 'Average Distance Travelled (km)', 'Daily Production/Quantity', 'Product Name', 'Category', 'Time of Harvest', 'Time to Loading', 'Source Location', 'Destination Location', 'Spoilage Percentage', 'Shelf Life') had zero importance in this specific Decision Tree model. This suggests that, at this limited depth, they did not contribute to any splits or were less predictive than the identified key features.

*Note: Feature importance for other models (e.g., coefficients for Logistic Regression/SGD, feature importance attributes for Random Forest/XGBoost) were not explicitly extracted in this notebook.*

---

### 3. Comparison and Contrast of Classification Models

**A. Performance (Accuracy, F1-score, Mean CV Accuracy):**
*   **Overall:** All models perform poorly, with accuracies hovering around 20%. This indicates a highly challenging multi-class classification problem.
*   **XGBoost (0.2090 Mean CV Accuracy):** Emerges as the slightly best-performing model in terms of generalization.
*   **Decision Tree and K-Nearest Neighbors (0.2020 Mean CV Accuracy):** Show very similar, slightly lower performance than XGBoost.
*   **Logistic Regression (0.2014 Mean CV Accuracy):** Comparable to DT and KNN.
*   **SVM, Random Forest, and SGD Classifier (0.1964 - 0.1998 Mean CV Accuracy):** These models perform marginally worse than the top group.

**B. Run Statistics (Training and Prediction Time):**
*   **Training Time:**
    *   **Fastest:** Logistic Regression (0.1084s), K-Nearest Neighbors (0.1130s), Decision Tree (0.2088s), and SGD Classifier (0.2615s) are very quick to train.
    *   **Moderate:** XGBoost (1.4667s) and Random Forest (1.7513s) take a bit longer due to their ensemble nature.
    *   **Slowest:** Support Vector Machine (3.0178s) is the most computationally intensive to train on this dataset, likely due to its complexity with many features.
*   **Prediction Time:** Most models have very fast prediction times (milliseconds), which is crucial for real-time applications. SGD Classifier has the fastest prediction time.

**C. Interpretability:**
*   **Decision Tree:** Highly interpretable due to its explicit decision rules and feature importance. The tree structure allows direct insight into how predictions are made and which features drive those decisions.
*   **Logistic Regression and SGD Classifier:** Offer some interpretability through coefficients, indicating the direction and strength of the relationship between features and the target. However, for multi-class problems and when many features are involved, this can still be complex.
*   **Random Forest and XGBoost:** These ensemble methods are less directly interpretable than a single Decision Tree, but they do provide feature importance scores, which can still offer valuable insights.
*   **Support Vector Machine and K-Nearest Neighbors:** Generally considered less interpretable, as their decision boundaries are often complex or instance-based (KNN).

**D. Model Appropriateness under different Considerations:**
*   **High Performance Potential (after tuning):** XGBoost, given its slightly superior performance, might be the best candidate for further optimization (hyperparameter tuning) if maximizing accuracy is the primary goal, despite its moderate training time.
*   **Speed and Baseline Performance:** Logistic Regression, Decision Tree, and KNN offer a good balance of speed and baseline performance, making them suitable for quick initial analyses or scenarios where computational resources are limited.
*   **Interpretability:** Decision Tree is the clear winner for interpretability, allowing domain experts to understand the decision-making process.
*   **Computational Cost:** SVM, despite its theoretical power, might be less appropriate for larger datasets or real-time applications due to its higher training cost unless significant accuracy gains can be demonstrated after extensive tuning.

---

### 4. Comprehensive Summary of Observations and Findings

The analysis of supervised machine learning models for classifying 'Initial Quality Grade' has revealed several critical observations and findings:

**A. Overall Model Performance and Challenges:**
All evaluated classification models (Logistic Regression, SVM, SGD, Decision Tree, KNN, Random Forest, XGBoost) yielded consistently low accuracy and F1-scores, typically ranging from 19% to 21%. This suggests that the problem is either intrinsically difficult, the current feature set is not sufficiently informative, or the models are not optimally configured. The consistency in low performance across diverse algorithms (linear, tree-based, ensemble, distance-based) indicates that basic model application is insufficient.

**B. Data Characteristics Impeding Performance:**
*   **Target Variable Distribution:** While the 'Initial Quality Grade' target variable shows a relatively balanced distribution across its 5 classes, suggesting that class imbalance is not the primary cause of low performance.
*   **Feature Importance:** The Decision Tree model highlighted 'Vehicle Type', 'Transport Duration (hours)', 'Transportation Cost per Trip', 'Humidity Maintained (%)', 'Daily Demand', 'Wastage Cost', 'Total Supply Chain Time', and 'Storage Cost' as the most influential features. However, a significant number of other features were deemed to have zero importance in the limited-depth Decision Tree, suggesting they might not be contributing effectively in their current form or need transformation.
*   **Numeric Feature Issues:** Most numeric features exhibit skewed distributions and the presence of outliers. Furthermore, significant multicollinearity was observed (e.g., 'Daily Production/Quantity' with 'Daily Demand', and 'Transport Duration (hours)' with 'Total Supply Chain Time'). These issues can negatively impact models that assume linearity or normality, making parameter estimation unstable and reducing predictive power.
*   **Categorical Feature Complexity:** Some categorical features like 'Product Name', 'Source Location', and 'Destination Location' possess high cardinality, while others have very skewed distributions ('Transportation Method', 'Vehicle Type'). The initial preprocessing involved Label Encoding followed by One-Hot Encoding which, for high-cardinality features, can lead to a bloated feature space and potential sparsity issues.
*   **Missing Value Handling:** The 'Tracking Method' feature had approximately 50% missing values, which were mode-imputed. This simple imputation strategy might introduce bias or obscure potential signals, especially if the missingness itself holds information.

**C. Reasons for Low Model Accuracies:**
The pervasive low accuracies are likely a confluence of:
1.  **Suboptimal Feature Engineering/Representation:** Many features might not be captured in a way that maximizes their predictive potential (e.g., simple encoding of high-cardinality features, lack of derived features).
2.  **Unaddressed Data Quality Issues:** Skewed distributions, outliers, and multicollinearity, if not properly handled, can degrade model performance.
3.  **Lack of Hyperparameter Optimization:** Running models with default parameters, rather than tuning them to the specific dataset, almost always results in suboptimal performance.
4.  **Challenging Nature of the Problem:** It's possible that predicting 'Initial Quality Grade' precisely is inherently complex, requiring very nuanced features or more advanced modeling approaches.

**D. Overall Model Appropriateness:**
*   No single model currently stands out as "appropriate" given the low performance.
*   **XGBoost** showed the best cross-validated performance, suggesting it might be the most promising candidate for further optimization efforts.
*   **Decision Tree** offers the best interpretability, providing clear rules which could be valuable for understanding the driving factors of quality.
*   **Logistic Regression** and **KNN** offer efficiency for rapid prototyping or as baselines.

**E. Next Steps and Recommendations for Improvement:**

To significantly improve model performance and overcome the identified challenges, the following concrete next steps are recommended:

1.  **Advanced Feature Engineering and Selection:**
    *   **Derive New Features:** Create informative features by combining existing ones (e.g., ratios like `Wastage_Rate = Wastage Cost / Daily Production/Quantity`, `Spoilage_per_ShelfLife = Spoilage Percentage (%) / Shelf Life (Days)`).
    *   **Time-Based Features:** Extract temporal features from 'Time of Harvest' and 'Time to Loading' (e.g., hour of day, day of week, duration between harvest and loading).
    *   **Interaction Terms:** Explore interaction terms between highly important features.
    *   **Feature Selection:** Use statistical methods or model-based selection techniques to identify and remove redundant or uninformative features.

2.  **Refined Data Preprocessing:**
    *   **Sophisticated Encoding for High-Cardinality Categoricals:** Implement techniques like Target Encoding (with proper cross-validation to avoid leakage), Feature Hashing, or grouping rare categories for features like 'Product Name', 'Source Location', 'Destination Location', and 'Tracking Method'.
    *   **Robust Handling of Skewed Numeric Distributions:** Apply Box-Cox or Yeo-Johnson transformations more systematically to features showing strong skewness.
    *   **Advanced Outlier Treatment:** Implement more sophisticated outlier detection (e.g., Isolation Forest) and treatment methods (e.g., Winsorization or imputation) rather than simple capping or removal.
    *   **Re-evaluate Missing Value Imputation:** For 'Tracking Method', consider treating missingness as a distinct category or using predictive imputation if the missing data mechanism is informative.

3.  **Systematic Hyperparameter Tuning:**
    *   **Grid Search / Randomized Search:** Employ `GridSearchCV` or `RandomizedSearchCV` for all models to thoroughly explore their hyperparameter spaces. This is critical for optimizing each model's configuration for the specific dataset.
    *   **Focus on Key Parameters:** Prioritize tuning parameters known to have a significant impact on model performance (e.g., `max_depth` for trees, `C` for SVM/LR, `n_estimators` for ensembles, `n_neighbors` for KNN).

4.  **Consistent and Robust Cross-Validation:**
    *   **Stratified K-Fold:** Continue to use Stratified K-Fold Cross-Validation for all model evaluation and hyperparameter tuning to ensure robust and unbiased performance estimates, especially for the multi-class target.
    *   **Pipelines for Preprocessing:** Integrate preprocessing steps (scaling, encoding, transformations) into `sklearn.pipeline.Pipeline` objects, especially when performing cross-validation, to prevent data leakage and ensure that each fold's data is preprocessed independently.

By implementing these comprehensive next steps, there is a strong potential to significantly improve the predictive accuracy and generalization capabilities of the supervised machine learning models for 'Initial Quality Grade'.


## Conclusion:

### Data Analysis Key Findings

*   **XGBoost Classifier** exhibited the highest mean cross-validation accuracy at approximately 0.2090.
*   **K-Nearest Neighbors** followed closely with a mean cross-validation accuracy of approximately 0.2020.
*   **Decision Tree** showed a mean cross-validation accuracy of approximately 0.2020, very similar to KNN.
*   **Logistic Regression** had a mean cross-validation accuracy of approximately 0.2014.
*   **Support Vector Machine (SVM)** achieved a mean cross-validation accuracy of approximately 0.1998.
*   **Random Forest Classifier** had a mean cross-validation accuracy of approximately 0.1972.
*   **SGD Classifier** yielded the lowest mean cross-validation accuracy among the evaluated models at approximately 0.1964.
*   The overall model performance comparison, updated with mean cross-validation accuracies, ranks XGBoost Classifier as the top performer, indicating better generalization ability compared to other models in this specific evaluation.

### Insights or Next Steps

*   The consistently low accuracy scores (around 20%) across all models suggest that the problem might be highly challenging or that the current features are not sufficiently discriminative. Further feature engineering or exploration of more complex model architectures could be beneficial.
*   Given that XGBoost Classifier, K-Nearest Neighbors, and Decision Tree performed marginally better in cross-validation, these models could be prioritized for hyperparameter tuning and more in-depth analysis to potentially improve their performance.

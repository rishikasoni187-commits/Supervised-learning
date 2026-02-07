# Supervised-learning
**Project Title** - *“Predictive Modeling Using Supervised Machine Learning Techniques for Classification and Decision Support”*
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

# README - Optimizing Food Delivery Logistics with Machine Learning

## Overview
This assignment focuses on addressing key logistical challenges in food delivery using data science and machine learning techniques. By leveraging a real-world dataset and applying predictive modeling, cluster analysis, and optimization methods, the goal is to provide actionable insights to improve delivery operations and enhance customer satisfaction.

---

## Steps Followed in the Assignment

### 1. Dataset Preparation
- **Dataset Selected:** Daily courier partner activity.
- **Cleaning:** Removed outliers, filled missing values, and ensured data consistency.
- **Feature Engineering:** Added new features like day of the week, season, and weather conditions.

### 2. Exploratory Data Analysis (EDA)
- **Visualization:** Analyzed trends in temperature, precipitation, and courier activity using graphs.
- **Insights:** Identified seasonal patterns and correlations impacting courier availability.

### 3. Feature Selection and Statistical Analysis
- **Approach:** Used Random Forest to determine feature importance.
- **Key Features:** Temperature, relative humidity, precipitation, and day-of-week variables.

### 4. Model Training and Optimization
- **Models Used:** Linear Regression, Random Forest Regressor, and Support Vector Machine (SVM).
- **Optimization:** Fine-tuned hyperparameters for Random Forest and selected the best-performing model based on RMSE and R² scores.

### 5. Cluster Analysis
- **Method:** KMeans clustering was applied to identify patterns in delivery locations.
- **Insights:** Cluster centroids provided key information about weather and courier activity trends.

### 6. Predictions
- **Questions Answered:**
  - Predicted next-day order volumes.
  - Forecasted courier availability for future dates.
  - Identified high-demand delivery locations.
- **Results:** Models delivered practical accuracy, aiding better resource allocation.

---

## Results and Key Metrics
- **Linear Regression:** Best model for predicting courier availability with an RMSE of 7.58.
- **Clustering Analysis:** Identified meaningful delivery patterns with five clusters.
- **Revenue Predictions:** Predicted daily revenue trends based on courier activity.

---

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/lakshmisrinidhp/logistics-ml.git
   cd logistics-ml
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts
Run the scripts in the following order:
```bash
python dataset_preparation.py
python feature_engineering.py
python exploratory_data_analysis.py
python model_training.py
python predict_orders.py
python predict_delivery_locations.py
python predict_revenue.py
```

### Viewing Outputs
- Check the results and visualizations in the `outputs/` folder.
- Detailed insights are documented in `report.pdf`.

---

## Further Development
- Incorporate advanced machine learning models like Gradient Boosting or Neural Networks.
- Include real-time data integration for dynamic predictions.
- Expand analysis with additional external factors, such as traffic data.

---

## Contact
For any queries or clarifications, please reach out via email included in the application. Thank you for reviewing this assignment!
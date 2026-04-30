Exploratory Data Analysis (EDA) Project
Overview

This project focuses on performing Exploratory Data Analysis (EDA) on a health-related dataset. The goal is to understand the structure of the data, identify patterns, detect anomalies, and extract meaningful insights that can guide further machine learning tasks.

Objectives
Understand dataset structure and features
Handle missing and inconsistent data
Detect and treat outliers
Explore relationships between variables
Visualize key patterns and trends
Prepare data for modeling

Dataset:
Source: kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
Number of rows:  5,110
Number of columns: 12 features

Key Features:
age – Patient age
bmi – Body Mass Index
avg_glucose_level – Average glucose level
stroke – Target variable (0 = No, 1 = Yes)


Data Cleaning Steps:
Removed duplicate rows
Handled missing values
Corrected data types
Standardized categorical values

Exploratory Analysis
Univariate Analysis
Distribution plots for numerical features
Count plots for categorical features
Bivariate Analysis
Boxplots comparing features vs target
Correlation analysis
Key Insights
Higher age is associated with increased stroke risk
BMI shows moderate variation across classes
Glucose level has noticeable influence on target

Visualizations
Histograms
Boxplots
Heatmaps
Count plots
(All visualizations are available in the notebook)

Tools & Libraries
Python
pandas
numpy
matplotlib
seaborn

Conclusion
The dataset was successfully cleaned and explored. Key patterns and relationships were identified, providing a solid foundation for building predictive models.

Contributors
Precious and Dicxin
Group 2
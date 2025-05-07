# Customer Segmentation

## Overview
Customer segmentation is the process of dividing customers into distinct groups based on similar characteristics. This interactive Streamlit web app implements unsupervised learning techniques like K-Means and DBSCAN to perform customer segmentation on numeric data.

## Features

- 🧼 **Data Preprocessing**: 
  - Handle missing values (drop or mean imputation)
  - Remove duplicates
  - Remove outliers (IQR method)
  - Optionally scale features using `StandardScaler`

- 📊 **Clustering Models**:
  - K-Means with Elbow Method and Silhouette Score analysis
  - DBSCAN with adjustable `eps` and `min_samples`

- 📈 **Interactive Visualizations**:
  - Cluster scatter plots with selectable axes
  - Cluster statistics (count, mean, std, min, max)
  - Cluster distribution percentages

- 📁 **User Upload**:
  - Upload your own CSV file
  - Select which numeric columns to use for clustering

- ⚠️ **Note**: This app currently supports **numeric features only** for clustering.

## Usage
1. Run the app using `streamlit run app.py`.
2. Upload a CSV file.
3. Select the columns and preprocessing options.
4. Choose a clustering method and visualize the results.

## Dependencies
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

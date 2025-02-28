import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Streamlit App Title
st.title("Customer Segmentation using K-Means and DBSCAN")
st.markdown("<h3 style='text-align: center;'>**It only handles Numeric Data**</h3>", unsafe_allow_html=True)
# Initialize session state variables
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "original_data" not in st.session_state:
    st.session_state.original_data = None
if "preprocessing_done" not in st.session_state:
    st.session_state.preprocessing_done = False
if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None
if "clustering_done" not in st.session_state:
    st.session_state.clustering_done = False

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.write(df.head())

        # Column Selection
        columns = st.multiselect("Select columns for clustering:", df.columns, default=df.columns[1:])

        if columns:
            original_data = df[columns].copy()
            data = original_data.copy()
            
            # Preprocessing Options
            st.write("### Preprocessing Options")
            scale_data = st.checkbox("Scale Data (StandardScaler)")
            remove_missing = st.checkbox("Remove Missing Values")
            fill_missing = st.checkbox("Fill Missing Values (Mean Imputation)")
            remove_outliers = st.checkbox("Remove Outliers (IQR)")
            remove_duplicate = st.checkbox("Remove Duplicate Rows")

            # Button to perform preprocessing
            if st.button("Perform Preprocessing"):
                if remove_missing:
                    data = data.dropna()
                elif fill_missing:
                    data = data.fillna(data.mean())

                if remove_outliers:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

                if remove_duplicate:
                    data = data.drop_duplicates()

                data = data.reset_index(drop=True)

                if scale_data:
                    scaler = StandardScaler()
                    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=columns)
                    st.session_state.preprocessed_data = scaled_data
                    st.session_state.original_data = data
                    st.session_state.scaler = scaler
                else:
                    st.session_state.preprocessed_data = data
                    st.session_state.original_data = data
                    st.session_state.scaler = None

                st.session_state.preprocessing_done = True
                st.success("✅ Preprocessing completed! Now you can proceed with clustering.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Ensure preprocessing is complete before allowing clustering
if st.session_state.preprocessing_done:
    data = st.session_state.preprocessed_data
    original_data = st.session_state.original_data
    scaler = st.session_state.scaler

    # Clustering Method Selection
    method = st.radio("Choose Clustering Method:", ["K-Means", "DBSCAN"])

    # Elbow Method for K-Means
    if method == "K-Means":
        inertia = []
        silhouette_scores = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))

        plt.figure(figsize=(8, 4))
        plt.plot(k_range, inertia, marker='o', linestyle='--')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal K")
        st.pyplot(plt)

        plt.figure(figsize=(8, 4))
        plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score Analysis")
        st.pyplot(plt)

        k = st.slider("Select number of clusters (K):", min_value=2, max_value=10, value=3)
    elif method == "DBSCAN":
        eps = st.slider("Select Epsilon (eps) value:", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
        min_samples = st.slider("Select Minimum Samples:", min_value=2, max_value=10, value=5)

    # Clustering
    clustered_df = original_data.copy()
    if method == "K-Means":
        kmeans = KMeans(n_clusters=k, random_state=42)
        clustered_df['Cluster'] = kmeans.fit_predict(data)
         # K-means Statistics
        st.write("### Cluster Statistics")
        cluster_stats = clustered_df.groupby('Cluster').agg({
        'Cluster': 'count',
        **{col: ['mean', 'std', 'min', 'max'] for col in columns}
        }).round(2)
    
    # Rename count column
        cluster_stats = cluster_stats.rename(columns={'Cluster': 'Count'})
        st.write(cluster_stats)
    
    # Show percentage distribution
        cluster_dist = (clustered_df['Cluster'].value_counts(normalize=True) * 100).round(2)
        st.write("### Cluster Distribution (%)")
        st.write(cluster_dist)
        centroids = kmeans.cluster_centers_
        if scaler:
            centroids_original = scaler.inverse_transform(centroids)
        else:
            centroids_original = centroids

        # Scatter Plot of Clusters
        if len(columns) >= 2:
            st.write("### Choose Axes for Clustering Plot")
            x_axis = st.selectbox("Select X-axis:", columns, key="x_axis")
            y_axis = st.selectbox("Select Y-axis:", columns, key="y_axis")
            plt.clf()  # Clear the current figure

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(clustered_df[x_axis], clustered_df[y_axis], c=clustered_df['Cluster'], cmap='viridis', alpha=0.6)
            plt.scatter(centroids_original[:, columns.index(x_axis)], centroids_original[:, columns.index(y_axis)], c='red', marker='x', s=100, label='Centroids')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title("K-Means Clustering Result")
            plt.legend()
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(plt)
    
    elif method == "DBSCAN":
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clustered_df['Cluster'] = dbscan.fit_predict(data)
          # DBSCAN Statistics
        st.write("### Cluster Statistics")
        cluster_stats = clustered_df.groupby('Cluster').agg({
        'Cluster': 'count',
        **{col: ['mean', 'std', 'min', 'max'] for col in columns}
         }).round(2)
        # Rename count column and cluster -1 to "Noise"
        cluster_stats = cluster_stats.rename(columns={'Cluster': 'Count'})
        cluster_stats.index = cluster_stats.index.map(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
        st.write(cluster_stats)
    
    # Show percentage distribution
        cluster_dist = (clustered_df['Cluster'].value_counts(normalize=True) * 100).round(2)
        cluster_dist.index = cluster_dist.index.map(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
        st.write("### Cluster Distribution (%)")
        st.write(cluster_dist)
    
        if len(columns) >= 1:
            st.write("### Choose Axes for Clustering Plot")
            x_axis = st.selectbox("Select X-axis:", columns, key="x_axis")
            y_axis = st.selectbox("Select Y-axis:", columns, key="y_axis")
            
            plt.clf()
            plt.close('all')
            plt.figure(figsize=(8, 6))
    
    
            scatter = plt.scatter(clustered_df[x_axis], clustered_df[y_axis], c=clustered_df['Cluster'], cmap='viridis', alpha=0.6)
            plt.title("DBSCAN Clustering Result")
    
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend()
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(plt)

          
        
#  K-Means Clustering

## 1. Project Overview

This project implements **K-Means Clustering**, an unsupervised machine learning algorithm, to solve a clustering problem. The main objective is to group data points into distinct clusters based on their similarities. The notebook walks through the entire process, from loading and preprocessing the dataset to building, training, and evaluating the K-Means model. By running this notebook, users will gain insights into how K-Means works, how to preprocess data for clustering, and how to evaluate the quality of the resulting clusters using various metrics and visualizations.

### Dataset

The dataset used in this project is a synthetic dataset generated specifically for demonstrating K-Means clustering. It contains multiple features (dimensions) that describe different data points, which are then clustered into groups based on their proximity in feature space. Before applying the K-Means algorithm, the data undergoes preprocessing steps such as scaling to ensure that all features contribute equally to the clustering process.

### Machine Learning Methods

- **K-Means Clustering**: This is an unsupervised learning algorithm used to partition data into \(k\) clusters. Each cluster is defined by its centroid, and data points are assigned to the nearest centroid. The algorithm iteratively adjusts centroids until convergence is achieved.
  
- **Elbow Method**: This method is used to determine the optimal number of clusters by plotting the sum of squared distances from each point to its assigned cluster center and looking for an "elbow" point where adding more clusters does not significantly improve the fit.

### Notebook Overview

The notebook is structured as follows:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded from a CSV file or generated synthetically.
   - Data scaling (e.g., using `StandardScaler`) is applied to ensure that all features contribute equally to the clustering process.
   
2. **Model Building**:
   - The K-Means model is defined using `sklearn.cluster.KMeans`.
   - The number of clusters \(k\) is initially set based on domain knowledge or determined using methods like the Elbow Method.
   
3. **Model Training**:
   - The K-Means model is trained by fitting it to the preprocessed data.
   - Key parameters include the number of clusters \(k\), initialization method (`k-means++`), and the number of iterations for convergence.
   
4. **Evaluation**:
   - Metrics such as inertia (sum of squared distances) are used to evaluate how well the model has clustered the data.
   - The Elbow Method helps in determining the optimal number of clusters.
   
5. **Visualization**:
   - Visualizations include scatter plots of clustered data points and centroids.
   - Elbow plots are used for determining the optimal number of clusters.

## 2. Requirements

### Running Locally

To run this notebook locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/K-Means-Clustering.git
    cd K-Means-Clustering
    ```

2. **Set up a virtual environment**:

    Using `venv`:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    Or using `conda`:
    ```bash
    conda create --name ml-env python=3.8
    conda activate ml-env
    ```

3. **Install project dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
    Open the notebook (`3-K_Means_Clustering.ipynb`) in the Jupyter interface to run it.

### Running Online via MyBinder

You can also run this notebook online without installing any software by using MyBinder:

Click the MyBinder button below to launch an interactive Jupyter notebook environment:


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/K-Means-Clustering.git/HEAD?labpath=notebooks)

Once MyBinder loads:
1. Navigate to your notebook (`3-K_Means_Clustering_Projekt.ipynb`) in the file browser on the left.
2. Click on the notebook to open it.
3. Run all cells using "Run All" (`Shift + Enter` for individual cells).

By using MyBinder, you can explore and run all code cells without needing any local setup.

## 3. Reproducing Results

To reproduce results from this notebook:

1. Open `3-K_Means_Clustering_Projekt.ipynb` in Jupyter (locally or via MyBinder).
2. Execute all cells sequentially by selecting them and pressing `Shift + Enter`.
3. Ensure that all cells execute without errors.
4. Observe output results directly within Jupyter, including evaluation metrics and visualizations.

### Interpreting Results:

- **Inertia (Sum of Squared Distances)**: This metric shows how well-separated your clusters are; lower values indicate better-defined clusters.
- **Elbow Plot**: Helps determine the optimal number of clusters by showing where adding more clusters yields diminishing returns in reducing inertia.
- **Cluster Visualization**: Scatter plots show how well data points are grouped into their respective clusters, with centroids marked for each cluster.
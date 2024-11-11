#  K-Means Clustering

## 1. Project Overview

This project demonstrates how **K-Means Clustering**, a popular machine learning algorithm, can be applied to categorize universities in the United States into meaningful groups based on various features (e.g., number of applications received, accepted students, enrollment rates, etc.). By clustering universities, we can help users identify similarities between institutions, such as whether a university is public or private, or how selective it is in its admissions process.

 The notebook walks through the entire process, from loading and preprocessing the dataset to building, training, and evaluating the K-Means model. By running this notebook, users will gain insights into how K-Means works, how to preprocess data for clustering, and how to evaluate the quality of the resulting clusters using various metrics and visualizations.

### Dataset

The dataset used in this project comes from a public educational database that contains data on various U.S. colleges and universities.

### Machine Learning Methods

- **K-Means Clustering**: K-Means is an unsupervised machine learning algorithm that divides data into **K clusters**. The goal is to partition the universities into groups where universities within the same group are more similar to each other than to universities in other groups.

 The algorithm starts by randomly selecting **K** initial cluster centers (centroids).
- Each data point (university) is assigned to the nearest centroid based on a distance metric (usually Euclidean distance).
- The centroids are updated iteratively until the clusters stabilize, meaning universities no longer change groups.

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

Though K-Means is unsupervised, we can compare the clusters against known attributes (e.g., private vs public) to evaluate the clustering performance. 

In this project, the dataset is divided into **two clusters**, corresponding to public and private universities, based on the feature space provided.

1. **Confusion Matrix**

The confusion matrix shows how well the K-Means algorithm has categorized the universities into clusters. It compares the predicted cluster labels from the algorithm with the true labels (public vs. private universities).

From the confusion matrix below:


[[138 74]

[531 34]]


- **Row 1**: 138 universities were correctly identified as belonging to the first cluster (e.g., public universities), but 74 were incorrectly categorized.
- **Row 2**: 34 universities were correctly identified as belonging to the second cluster (e.g., private universities), while 531 were misclassified.

This indicates that the model is struggling to accurately distinguish between the two clusters.

2. **Classification Report**


- **Precision**: This tells us how many of the predicted universities in a given cluster actually belong to that cluster. A precision of 0.21 for cluster 0 means that only 21% of universities predicted to be in cluster 0 are indeed correctly categorized.
- **Recall**: This tells us how many of the true universities in a given cluster were correctly identified. A recall of 0.65 for cluster 0 means that 65% of universities in cluster 0 were correctly identified, but the recall for cluster 1 is much lower.
- **F1-Score**: The F1-score is the harmonic mean of precision and recall. A score of 0.31 for cluster 0 indicates a moderate performance, but the F1-score for cluster 1 is quite low (0.10), meaning the model is struggling to identify universities in that cluster.

### Accuracy

The overall accuracy of the model is **22%**, which indicates that the clustering is not very accurate. However, this can be expected with unsupervised methods like clustering, where labels are not provided during training.

## Conclusion

K-Means clustering helps to automatically categorize universities into meaningful groups based on their characteristics. However, the initial model needs improvement, as shown by the confusion matrix and classification report results. By refining the input features and tuning the model, we can achieve more accurate and insightful clusters that can be useful.
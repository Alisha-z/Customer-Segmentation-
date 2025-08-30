# Customer-Segmentation-

# 🛍️ Customer Segmentation using Unsupervised Clustering

## 📌 Objective
The goal of this project is to **segment customers into meaningful groups** based on their spending behavior. By applying clustering algorithms, businesses can better understand their customers and tailor marketing strategies to different groups.

## 📂 Dataset
- **Mall Customers Dataset**  
  Contains customer demographic and spending-related attributes such as:
  - CustomerID  
  - Gender  
  - Age  
  - Annual Income  
  - Spending Score  

## ⚙️ Technologies & Libraries
- **Python 3.x**
- **NumPy** – numerical computations  
- **Pandas** – data handling and preprocessing  
- **Matplotlib / Seaborn** – data visualization  
- **Scikit-learn** – machine learning algorithms (K-Means, DBSCAN, PCA)  

## 🧠 Methods & Approach
1. **Data Preprocessing**
   - Handled missing values (if any)
   - Normalized features using `MinMaxScaler`

2. **Clustering**
   - Applied **K-Means clustering**  
   - Used the **Elbow Method** and **Silhouette Score** to find the optimal number of clusters  
   - Experimented with **DBSCAN** as an alternative  

3. **Dimensionality Reduction**
   - Applied **PCA (Principal Component Analysis)** to reduce dimensions for better visualization  

4. **Visualization**
   - 2D and 3D cluster plots  
   - Pair plots using Seaborn  

## 🎯 Learning Outcomes
- Understanding unsupervised learning concepts  
- Applying and evaluating clustering algorithms (K-Means, DBSCAN)  
- Using dimensionality reduction for visualization (PCA)  
- Interpreting customer groups for business insights  



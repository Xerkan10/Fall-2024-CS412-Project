# Fall-2024-CS412-Project

This project applies a combination of classification and regression techniques to analyze Instagram user data and post engagement. The work is divided into two primary tasks:

1. **Classification** of Instagram accounts into predefined categories based on text and numerical features.
2. **Regression** to predict like counts of individual posts using engineered features.

---

## 📊 Task 1: Classification of Instagram Accounts

### Objective
Classify Instagram accounts into distinct categories using a combination of text and numerical features.

### 🔧 Preprocessing and Feature Engineering

For the classification task, I implemented a robust preprocessing pipeline that became the basis for further improvements in subsequent rounds.

- **Text Preprocessing:** User biographies and post captions were cleaned by removing URLs, special characters (except hashtags and emojis), and numbers. Text was normalized to lowercase with special handling for Turkish-specific letters.

- **TF-IDF Vectorization:** The cleaned text was vectorized using TF-IDF, generating 7,500 features. No dimensionality reduction was applied in Round 1, which increased computation and introduced potential noise.

- **Numerical Features:** I included metadata such as `follower_count`, `following_count`, and `average_like` to complement textual features. Although correlations were weak, they contributed additional context.

- **Stratified Train-Test Split:** To address class imbalance, a stratified split ensured proportional representation across training and test sets.

### ⚙️ Model and Evaluation

- **Model Used:** Stacking Classifier (ensemble of multiple base learners with a meta-classifier).
- **Validation Accuracy:** ~0.65
- **Competition Performance:**
  - **Accuracy:** 0.7832 (Ranked 37/141)
  - **F1-Weighted Score:** 0.7733 (Ranked 40/141)

### 🔍 Observations and Insights

- The large TF-IDF feature space introduced noise; dimensionality reduction would likely improve performance.
- Hyperparameter tuning was minimal in Round 1, and optimizing base/meta-models could lead to better outcomes.
- Numerical features added some value, but couldn't compensate for the noisy textual input.

### 🚀 Round 2 Improvements: Classification

Building upon Round 1, I refined both preprocessing and modeling steps to enhance performance in Round 2.

#### 🔧 Preprocessing & Feature Engineering

- **Text Preprocessing:** Continued with URL removal, special character filtering (preserving emojis and hashtags), and tokenization. Applied TF-IDF with 7,500 features.
- **Correlation Analysis:** Created a correlation matrix to identify feature relationships. For example:
  - `is_verified` & `average_like`: **+0.269**
  - `is_business_account` & `average_like`: **−0.189**
- **Dimensionality Reduction:**
  - **PCA:** Found that 500 components explained ~50% variance.
  - **Truncated SVD:** Reduced text features to 1,000 dimensions for improved efficiency.

#### ⚙️ Model and Training

- **Model:** Enhanced **Stacking Classifier** with tuned base estimators:
  - Gradient Boosting, Logistic Regression (L1/L2), SVM (RBF, C=50), MLP
  - **Final Estimator:** Random Forest (max_depth=50, min_samples_leaf=2)
- **Validation:** 5-fold cross-validation for reliable accuracy estimation (~65%).

#### 📊 Results

- **Competition Accuracy:** **82.6%**
- **F1-Weighted Score:** **0.819**
- **Leaderboard Position:** **Rank 51/138 (Accuracy)**, **Rank 52/138 (F1)**


---

## 📈 Task 2: Like Count Prediction

### Objective
Predict the number of likes an Instagram post would receive, using both time-based and account-based features.

### 🔧 Feature Engineering

- **Feature Selection:** I iteratively tested different combinations of features, as feature selection proved critical to performance. Some features that initially seemed promising actually decreased performance due to noise or redundancy.

- **Key Features Used:**
  - Classical metrics: `follower_count`, `following_count`, `average_like_count`
  - Time-based features extracted from `timestamp`: 
    - `day_of_week`, `day_of_year`, and 
    - Time of day segments: `morning`, `afternoon`, `evening`, `night`
  - A potentially useful feature, `until_now_average_count`, was considered but dropped due to insufficient post history per user.

- **Missing Data Handling:** All null feature values were filled with zeros to ensure consistency.

### ⚙️ Model and Evaluation

- **Random Forest Regressor:**
  - **Log MSE on validation split:** ~0.48
  - Served as the baseline model with stable results.
  
- **Support Vector Machine (SVM):**
  - Performance was poor and computational time was high.

---

## 💡 Key Takeaways

- Effective **feature engineering** and **dimensionality reduction** are essential when working with high-dimensional textual data.
- **Hyperparameter optimization** plays a crucial role in improving model performance, especially in ensemble methods.
- Time-based features can provide valuable insights into post engagement and should not be overlooked.

---

## 🛠️ Tools and Libraries Used

- Python, Scikit-learn, Pandas, NumPy
- TF-IDF Vectorization
- Random Forest, SVM, Stacking Classifier

---




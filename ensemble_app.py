"""
Ensemble Learning Techniques - Interactive Tutorial
SEAS-8505 - Applied Machine Intelligence and Reinforcement Learning
Instructor: Dr. Elbasheer
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
                              RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="SEAS-8505 Ensemble Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
    }

    /* Section headers */
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 20px;
    }

    h2 {
        color: #34495e;
        background: linear-gradient(90deg, #e8f4f8 0%, #ffffff 100%);
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    h3 {
        color: #2980b9;
        padding: 8px 0;
        border-bottom: 2px solid #ecf0f1;
        margin-top: 20px;
    }

    h4 {
        color: #16a085;
        font-weight: 600;
        margin-top: 15px;
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.05rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e0e7ff 0%, #b8c5f2 100%);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #5568d3;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #e8f5e9 0%, #ffffff 100%);
        border: 2px solid #4caf50;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #2e7d32;
        padding: 12px 20px;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #c8e6c9 0%, #e8f5e9 100%);
        border-color: #388e3c;
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    /* Section dividers */
    hr {
        margin: 30px 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #3498db, transparent);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #2c3e50;
        font-weight: 700;
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Custom info box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 20px 0;
    }

    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    """Load the IBM Attrition dataset and prepare it for modeling."""
    df = pd.read_csv('myIBMAttrition.csv')

    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Split data into training and testing sets
    # test_size=0.2 means 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features to have similar ranges
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, df

# Helper function to display results
def display_results(y_test, predictions, title):
    """Display model performance metrics and confusion matrix."""
    accuracy = accuracy_score(y_test, predictions)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.text("Classification Report:")
        report = classification_report(y_test, predictions)
        st.text(report)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# Load data
X_train, X_test, y_train, y_test, df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150/1f77b4/ffffff?text=ML", width=150)
    st.markdown("### Course Information")
    st.info("**SEAS-8505**\nApplied Machine Intelligence\nand Reinforcement Learning")
    st.success("**Instructor**\nDr. Elbasheer")

    st.markdown("---")
    st.markdown("### About This App")
    st.markdown("""
    This interactive application demonstrates five ensemble learning techniques:

    - **Voting**
    - **Bagging**
    - **Boosting**
    - **Stacking**
    - **Blending**

    Each tab includes explanations, code examples, and live results.
    """)

    st.markdown("---")
    st.markdown("### Dataset Info")
    st.metric("Total Samples", len(df))
    st.metric("Features", len(df.columns) - 1)
    st.metric("Classes", len(df['Attrition'].unique()))

# Main title with custom styling
st.markdown('<div class="main-header">Ensemble Learning Techniques</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SEAS-8505: Applied Machine Intelligence and Reinforcement Learning | Instructor: Dr. Elbasheer</div>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs for different techniques
tabs = st.tabs([
    "Overview",
    "Voting",
    "Bagging",
    "Boosting",
    "Stacking",
    "Blending",
    "Popular Models"
])

# ==================== OVERVIEW TAB ====================
with tabs[0]:
    st.markdown("## 📚 What is Ensemble Learning?")

    # Introduction box
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #1976d2; margin-top: 0;'>🎯 Core Concept</h3>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
        Ensemble learning combines multiple machine learning models to create a stronger predictor.
        Think of it like asking multiple experts for their opinion before making a decision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 💡 Why Use Ensemble Methods?

        <div class='success-box'>
        <ul style='font-size: 1.05rem; line-height: 1.8;'>
            <li><strong>Better Accuracy:</strong> Multiple models often outperform a single model</li>
            <li><strong>Reduced Overfitting:</strong> Combining models helps avoid learning noise in the data</li>
            <li><strong>More Robust:</strong> Less sensitive to peculiarities in the training data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 📊 Dataset Information

        <div class='warning-box'>
        <p style='font-size: 1.05rem; line-height: 1.6;'>
        We are using the <strong>IBM Employee Attrition dataset</strong> to predict whether an employee
        will leave the company. The goal is to predict whether an employee will <strong>stay (0)</strong>
        or <strong>leave (1)</strong> based on various features.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Samples", len(df), delta="Complete Dataset")
        st.metric("Features", len(df.columns) - 1, delta="Input Variables")
        st.metric("Target Classes", len(df['Attrition'].unique()), delta="Binary Classification")

        st.markdown("### 📉 Class Distribution")
        attrition_counts = df['Attrition'].value_counts()
        st.metric("Stayed (0)", attrition_counts[0], delta=f"{attrition_counts[0]/len(df)*100:.1f}%")
        st.metric("Left (1)", attrition_counts[1], delta=f"{attrition_counts[1]/len(df)*100:.1f}%")

    st.markdown("---")

    st.markdown("---")
    # Add ensemble methods overview diagram
    st.markdown("## 🔄 Ensemble Methods Overview")
    try:
        st.image("overview-ensemble-method.png", width=700,
                caption="Overview of Different Ensemble Learning Techniques")
    except:
        st.info("Diagram not found. Place 'overview-ensemble-method.png' in the app directory.")

    st.markdown("---")

    with st.expander("📋 View Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("💻 View Data Preparation Code"):
        st.code("""
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('myIBMAttrition.csv')

# Separate features (X) and target variable (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features to have mean=0 and standard deviation=1
# This helps models perform better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
        """, language="python")

    # Key Concepts Section - Overview Tab Only
    st.markdown("---")
    st.markdown("## Key Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Ensemble Techniques Summary

        **Voting**
        - Simple combination of different algorithms
        - Good when you have diverse models
        - Easy to implement and interpret

        **Bagging**
        - Multiple models on random subsets
        - Reduces variance (overfitting)
        - Can be parallelized for speed

        **Boosting**
        - Sequential models that correct previous errors
        - Reduces bias (underfitting)
        - Often gives best performance
        """)

    with col2:
        st.markdown("""
        ### Advanced Techniques

        **Stacking**
        - Uses cross-validation to train a meta-learner
        - Combines diverse models intelligently
        - More complex but powerful

        **Blending**
        - Simpler alternative to stacking
        - Uses holdout validation set
        - Faster training time

        ### Best Practices
        - Start with simple methods (Voting, Bagging)
        - Try Boosting for better performance
        - Use Stacking/Blending for competitions
        """)

# ==================== POPULAR MODELS TAB ====================
with tabs[6]:
    st.markdown("## 🌟 Popular Ensemble Models")

    st.markdown("""
    <div class='info-box'>
    <p style='font-size: 1.1rem; line-height: 1.6;'>
    This section covers the most widely-used ensemble models in machine learning practice.
    These models have become industry standards due to their robust performance and versatility.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Random Forest Section
    st.markdown("---")
    st.markdown("## 🌲 1. Random Forest")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Theoretical Foundation

        Random Forest is a **bagging-based ensemble** that combines multiple decision trees to create
        a more robust and accurate model. Proposed by Leo Breiman in 2001, it addresses the high
        variance problem of individual decision trees.

        **Key Concepts:**

        **Bootstrap Aggregating (Bagging):**
        - Creates multiple subsets of training data through random sampling with replacement
        - Each tree is trained on a different bootstrap sample
        - Reduces variance by averaging predictions from multiple trees

        **Feature Randomness:**
        - At each split, only a random subset of features is considered
        - Typical subset size: √n for classification, n/3 for regression (n = total features)
        - This decorrelates the trees, making the ensemble more diverse

        **Ensemble Prediction:**
        - Classification: Majority voting from all trees
        - Regression: Average of all tree predictions
        """)

    with col2:
        st.markdown("""
        ### Mathematical Formulation

        **For Classification:**
        """)
        st.latex(r"\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_B(x)\}")

        st.markdown("**For Regression:**")
        st.latex(r"\hat{y} = \frac{1}{B}\sum_{i=1}^{B} h_i(x)")

        st.markdown("""
        Where:
        - B = number of trees
        - h_i(x) = prediction from tree i
        - x = input features
        """)

    st.markdown("""
    ### Advantages and Limitations

    **Advantages:**
    - High accuracy on most problems without extensive tuning
    - Resistant to overfitting when number of trees is large
    - Handles missing values and maintains accuracy with missing data
    - Provides feature importance rankings
    - Works well with both categorical and numerical features
    - Naturally handles multi-class classification
    - Can model non-linear relationships

    **Limitations:**
    - Less interpretable than single decision trees
    - Memory intensive for large forests
    - Slower prediction time compared to linear models
    - Can overfit on noisy datasets
    - Biased toward features with more categories
    - Not suitable for extrapolation beyond training data range

    ### Practical Applications

    **Industry Use Cases:**
    - **Finance**: Credit scoring, fraud detection, algorithmic trading
    - **Healthcare**: Disease diagnosis, patient readmission prediction
    - **E-commerce**: Customer churn prediction, recommendation systems
    - **Manufacturing**: Quality control, predictive maintenance
    - **Marketing**: Customer segmentation, response modeling

    ### Hyperparameter Tuning

    **Critical Parameters:**
    - `n_estimators`: Number of trees (more is usually better, but with diminishing returns)
    - `max_depth`: Maximum depth of each tree (controls complexity)
    - `min_samples_split`: Minimum samples required to split a node
    - `min_samples_leaf`: Minimum samples required at leaf nodes
    - `max_features`: Number of features to consider for each split
    - `bootstrap`: Whether to use bootstrap samples

    **Tuning Strategy:**
    - Start with defaults and increase n_estimators
    - Tune max_depth to balance bias-variance
    - Adjust min_samples_split and min_samples_leaf to prevent overfitting
    """)

    with st.expander("💻 View Random Forest Implementation"):
        st.code("""
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Initialize Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    max_depth=10,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples at leaf node
    max_features='sqrt',     # Number of features for best split
    bootstrap=True,          # Use bootstrap sampling
    random_state=42,         # For reproducibility
    n_jobs=-1               # Use all processors
)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)
print(f"\\nCross-validation Score: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 Most Important Features:")
print(feature_importance.head(10))
        """, language="python")

    # Gradient Boosting Section
    st.markdown("---")
    st.markdown("## ⚡ 2. Gradient Boosting Machines (GBM)")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Theoretical Foundation

        Gradient Boosting is a **sequential ensemble** technique that builds models iteratively,
        where each new model corrects errors made by previous models. It was popularized by
        Jerome Friedman in 1999.

        **Key Concepts:**

        **Sequential Learning:**
        - Models are added one at a time
        - Each new model focuses on correcting residual errors
        - Unlike bagging, trees are not independent

        **Gradient Descent in Function Space:**
        - Optimizes a loss function by iteratively adding models
        - Each new model is fit to the negative gradient of the loss
        - Moves toward the optimal prediction by taking small steps

        **Residual Fitting:**
        - First model makes initial predictions
        - Subsequent models predict the residuals (errors)
        - Final prediction is the sum of all model predictions

        **Shrinkage (Learning Rate):**
        - Controls the contribution of each tree
        - Smaller values require more trees but often improve generalization
        - Acts as regularization to prevent overfitting
        """)

    with col2:
        st.markdown("""
        ### Mathematical Formulation

        **Iterative Model Building:**
        """)
        st.latex(r"F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)")

        st.markdown("**Final Prediction:**")
        st.latex(r"\hat{y} = F_M(x) = \sum_{m=1}^{M} \nu \cdot h_m(x)")

        st.markdown("""
        Where:
        - F_m(x) = ensemble after m iterations
        - ν = learning rate (shrinkage)
        - h_m(x) = weak learner at iteration m
        - M = total number of iterations
        """)

    st.markdown("""
    ### Advantages and Limitations

    **Advantages:**
    - Often achieves state-of-the-art performance
    - Handles mixed data types well
    - Supports custom loss functions
    - Provides feature importance
    - Robust to outliers (with appropriate loss functions)
    - Can model complex non-linear relationships
    - Better at handling imbalanced datasets than Random Forest

    **Limitations:**
    - Sensitive to hyperparameter tuning
    - Prone to overfitting with too many iterations
    - Longer training time (sequential process)
    - Requires careful validation to prevent overfitting
    - Less parallelizable than Random Forest
    - Can be unstable with noisy data

    ### Practical Applications

    **Industry Use Cases:**
    - **Web Search**: Ranking search results (Yahoo, Bing)
    - **Computer Vision**: Object detection and classification
    - **Kaggle Competitions**: Winning solution in many competitions
    - **Risk Assessment**: Insurance pricing, loan default prediction
    - **Ecology**: Species distribution modeling
    - **Physics**: Particle identification in high-energy physics

    ### Hyperparameter Tuning

    **Critical Parameters:**
    - `n_estimators`: Number of boosting rounds (more trees)
    - `learning_rate`: Shrinkage parameter (0.01 to 0.3 typical)
    - `max_depth`: Depth of each tree (3-8 typical, shallow trees work well)
    - `subsample`: Fraction of samples for fitting trees (0.5-1.0)
    - `min_samples_split`: Minimum samples to split nodes
    - `min_samples_leaf`: Minimum samples at leaf nodes

    **Tuning Strategy:**
    - Start with small learning_rate (0.1) and tune n_estimators
    - Keep trees shallow (max_depth 3-5)
    - Use subsample < 1.0 for stochastic gradient boosting
    - Monitor validation performance to detect overfitting
    - Balance learning_rate and n_estimators (inverse relationship)
    """)

    with st.expander("💻 View Gradient Boosting Implementation"):
        st.code("""
# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Initialize Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contribution of each tree
    max_depth=3,             # Maximum depth of trees (keep shallow)
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples at leaf
    subsample=0.8,           # Fraction of samples for fitting (stochastic GB)
    random_state=42,
    verbose=0                # Set to 1 to see training progress
)

# Train the model
gb_clf.fit(X_train, y_train)

# Make predictions
y_pred = gb_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(gb_clf, X_train, y_train, cv=5)
print(f"\\nCross-validation Score: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Feature importance
print("\\nTop 10 Most Important Features:")
importances = gb_clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Staged predictions (useful for finding optimal n_estimators)
staged_accuracy = []
for y_pred_stage in gb_clf.staged_predict(X_test):
    staged_accuracy.append(accuracy_score(y_test, y_pred_stage))

print(f"\\nOptimal number of estimators: {np.argmax(staged_accuracy) + 1}")
print(f"Best staged accuracy: {max(staged_accuracy):.2%}")
        """, language="python")

    # XGBoost Section
    st.markdown("---")
    st.markdown("## 🚀 3. XGBoost (Extreme Gradient Boosting)")

    st.markdown("""
    ### Theoretical Foundation

    XGBoost is an optimized implementation of gradient boosting developed by Tianqi Chen in 2014.
    It has become the go-to algorithm for structured data in machine learning competitions and industry.

    **Advanced Features:**

    **Regularization:**
    - L1 (Lasso) and L2 (Ridge) regularization on leaf weights
    - Helps prevent overfitting better than traditional GBM
    - Controlled by `alpha` (L1) and `lambda` (L2) parameters

    **Tree Pruning:**
    - Uses "max_depth" first, then prunes trees backward
    - Removes splits that don't provide gain beyond a threshold
    - More efficient than traditional greedy approaches

    **Handling Missing Values:**
    - Learns the best direction to handle missing values
    - No need for imputation
    - Automatically learns optimal default directions

    **Parallel Processing:**
    - Parallelized tree construction
    - Column block structure for parallel learning
    - Cache-aware access patterns for better performance

    **Built-in Cross-Validation:**
    - Can perform cross-validation at each iteration
    - Early stopping based on validation performance
    - Helps prevent overfitting automatically

    ### Advantages over Traditional GBM

    - **Speed**: 10x faster than traditional implementations
    - **Performance**: Often achieves better accuracy
    - **Regularization**: Built-in L1/L2 regularization
    - **Flexibility**: Custom loss functions and evaluation metrics
    - **Scalability**: Handles large datasets efficiently
    - **Missing Values**: Native handling without imputation

    ### Practical Applications

    **Dominant in:**
    - Kaggle competitions (used in majority of winning solutions)
    - Click-through rate prediction
    - Customer behavior modeling
    - Risk modeling in finance
    - Medical diagnosis systems
    - Recommender systems
    """)

    with st.expander("💻 View XGBoost Implementation"):
        st.code("""
# Import necessary libraries
# Install XGBoost first: pip install xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Initialize XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,        # Number of boosting rounds
    learning_rate=0.1,       # Step size shrinkage
    max_depth=5,             # Maximum tree depth
    min_child_weight=1,      # Minimum sum of instance weight in child
    gamma=0,                 # Minimum loss reduction for split
    subsample=0.8,           # Subsample ratio of training instances
    colsample_bytree=0.8,    # Subsample ratio of columns
    reg_alpha=0,             # L1 regularization term
    reg_lambda=1,            # L2 regularization term
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train with early stopping
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose=False
)

# Make predictions
y_pred = xgb_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5)
print(f"\\nCross-validation Score: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Feature importance (multiple methods)
print("\\nFeature Importance (Gain):")
importance_gain = xgb_clf.get_booster().get_score(importance_type='gain')
print(sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)[:10])

# Get the best iteration
print(f"\\nBest iteration: {xgb_clf.best_iteration}")
print(f"Best score: {xgb_clf.best_score:.4f}")
        """, language="python")

    # AdaBoost Section
    st.markdown("---")
    st.markdown("## 🎯 4. AdaBoost (Adaptive Boosting)")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Theoretical Foundation

        AdaBoost, proposed by Freund and Schapire in 1996, was one of the first successful
        boosting algorithms. It won the Gödel Prize in 2003 for its theoretical contributions.

        **Key Concepts:**

        **Adaptive Weighting:**
        - Initially, all samples have equal weight
        - After each iteration, misclassified samples get higher weights
        - Next model focuses more on previously misclassified samples

        **Weak Learner Combination:**
        - Uses simple models (often decision stumps)
        - Each weak learner only needs to be slightly better than random
        - Combines weak learners into a strong classifier

        **Sequential Weight Update:**
        - Calculates error rate for each weak learner
        - Assigns weight based on accuracy
        - More accurate learners have higher influence

        **Final Prediction:**
        - Weighted majority vote of all weak learners
        - Weights depend on individual learner accuracy
        """)

    with col2:
        st.markdown("""
        ### Mathematical Formulation

        **Weight Update:**
        """)
        st.latex(r"w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t \cdot I(y_i \neq h_t(x_i))}")

        st.markdown("**Learner Weight:**")
        st.latex(r"\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)")

        st.markdown("**Final Classifier:**")
        st.latex(r"H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)")

        st.markdown("""
        Where:
        - w_i = weight of sample i
        - α_t = weight of learner t
        - ε_t = error rate of learner t
        - h_t(x) = prediction of learner t
        """)

    st.markdown("""
    ### Advantages and Limitations

    **Advantages:**
    - Simple to implement and understand
    - Works well with weak learners
    - No need to tune many parameters
    - Provides variable importance
    - Can achieve high accuracy with simple base learners
    - Less prone to overfitting than some methods

    **Limitations:**
    - Sensitive to noisy data and outliers
    - Can overfit if too many iterations
    - Performance depends on quality of weak learners
    - Slower than some modern alternatives
    - May struggle with very complex problems

    ### Practical Applications

    - Face detection (Viola-Jones algorithm)
    - Text classification
    - Bioinformatics (gene classification)
    - Customer churn prediction
    - Medical diagnosis
    """)

    with st.expander("💻 View AdaBoost Implementation"):
        st.code("""
# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Initialize AdaBoost with Decision Stumps (max_depth=1)
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,         # Number of boosting iterations
    learning_rate=1.0,       # Weight applied to each classifier
    algorithm='SAMME.R',     # Real AdaBoost (uses probabilities)
    random_state=42
)

# Train the model
ada_clf.fit(X_train, y_train)

# Make predictions
y_pred = ada_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(ada_clf, X_train, y_train, cv=5)
print(f"\\nCross-validation Score: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Estimator weights (importance of each weak learner)
print("\\nEstimator Weights (first 10):")
print(ada_clf.estimator_weights_[:10])

# Estimator errors
print("\\nEstimator Errors (first 10):")
print(ada_clf.estimator_errors_[:10])

# Feature importance
print("\\nFeature Importance:")
print(ada_clf.feature_importances_[:10])
        """, language="python")

    # Comparison Table
    st.markdown("---")
    st.markdown("## 📊 Comparative Analysis")

    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'AdaBoost'],
        'Type': ['Bagging', 'Boosting', 'Boosting', 'Boosting'],
        'Training': ['Parallel', 'Sequential', 'Sequential (optimized)', 'Sequential'],
        'Speed': ['Fast', 'Moderate', 'Fast', 'Moderate'],
        'Accuracy': ['High', 'Very High', 'Very High', 'Moderate-High'],
        'Overfitting Risk': ['Low', 'Moderate-High', 'Low-Moderate', 'Moderate'],
        'Tuning Difficulty': ['Easy', 'Moderate', 'Moderate-Hard', 'Easy'],
        'Interpretability': ['Low', 'Low', 'Low', 'Moderate']
    })

    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("""
    ### When to Use Which Model

    **Random Forest:**
    - Need fast training and prediction
    - Want good performance without much tuning
    - Have computational resources for parallel processing
    - Need feature importance quickly

    **Gradient Boosting:**
    - Need highest possible accuracy
    - Have time for hyperparameter tuning
    - Data is relatively clean
    - Can afford longer training time

    **XGBoost:**
    - Working on structured/tabular data
    - Competing in data science competitions
    - Need best performance with regularization
    - Have missing values in data

    **AdaBoost:**
    - Working with simple classification problems
    - Need interpretable weak learners
    - Want simple implementation
    - Have clean data without outliers
    """)

# ==================== VOTING TAB ====================
with tabs[1]:
    st.markdown("## 🗳️ Voting Classifier")

    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #1976d2; margin-top: 0;'>How Voting Works</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Voting combines predictions from multiple different algorithms. Each model gets a "vote" on the final prediction.
    Think of it like a panel of judges scoring a competition.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(90deg, #fff3e0 0%, #ffffff 100%); padding: 15px; border-radius: 8px; margin: 15px 0;'>
    <ul style='font-size: 1.05rem; line-height: 1.8; margin: 0;'>
        <li><strong>Hard Voting:</strong> Each model votes for a class, and the majority wins</li>
        <li><strong>Soft Voting:</strong> Each model provides probabilities, and we average them</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Visual Diagrams")

    # Visual diagrams for voting
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔵 Hard Voting")
        try:
            st.image("hard-voting.png", width=500,
                    caption="Hard Voting: Majority class wins")
        except:
            st.info("Hard voting diagram not found")

    with col2:
        st.markdown("### 🟢 Soft Voting")
        try:
            st.image("soft-voting-unweighted.png", width=500,
                    caption="Soft Voting: Average probabilities (Unweighted)")
        except:
            st.info("Soft voting diagram not found")

    # Weighted soft voting
    st.markdown("### 🟡 Weighted Soft Voting")
    try:
        st.image("soft-voting-weighted.png", width=600,
                caption="Soft Voting: Weighted average of probabilities")
    except:
        pass

    st.markdown("---")
    with st.expander("💻 View Complete Implementation Code"):
        st.markdown("**Complete Voting Classifier Code**")
        st.code("""
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create Base Classifiers
# These are different types of models that will vote
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Step 2: Create Voting Classifier

# Option A - Hard Voting: Each model votes for a class
voting_clf_hard = VotingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    voting='hard'  # Use majority vote
)

# Option B - Soft Voting: Average the probability predictions
voting_clf_soft = VotingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    voting='soft'  # Use average of probabilities
)

# Step 3: Train the Voting Classifier
# Choose one: hard or soft
voting_clf = voting_clf_soft  # or voting_clf_hard
voting_clf.fit(X_train, y_train)

# Step 4: Make Predictions
predictions = voting_clf.predict(X_test)

# Step 5: Evaluate Performance
accuracy = accuracy_score(y_test, predictions)
print(f"Voting Classifier Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))
        """, language="python")

    st.markdown("---")

    voting_type = st.radio("Select Voting Type:", ["Hard Voting", "Soft Voting"])

    if st.button("Run Voting Classifier", type="primary"):
        with st.spinner("Training models..."):
            # Create three different classifiers
            # These are the "voters" in our ensemble
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=5)

            # Combine them into a voting classifier
            if voting_type == "Hard Voting":
                voting_clf = VotingClassifier(
                    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
                    voting='hard'  # Use majority vote
                )
            else:
                voting_clf = VotingClassifier(
                    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
                    voting='soft'  # Use average of probabilities
                )

            # Train the ensemble
            voting_clf.fit(X_train, y_train)

            # Make predictions
            predictions = voting_clf.predict(X_test)

            st.success("Training complete!")
            display_results(y_test, predictions, f"{voting_type} Results")

            # Show individual model performance for comparison
            st.subheader("Individual Model Performance")
            col1, col2, col3 = st.columns(3)

            with col1:
                dt.fit(X_train, y_train)
                dt_acc = accuracy_score(y_test, dt.predict(X_test))
                st.metric("Decision Tree", f"{dt_acc:.2%}")

            with col2:
                lr.fit(X_train, y_train)
                lr_acc = accuracy_score(y_test, lr.predict(X_test))
                st.metric("Logistic Regression", f"{lr_acc:.2%}")

            with col3:
                knn.fit(X_train, y_train)
                knn_acc = accuracy_score(y_test, knn.predict(X_test))
                st.metric("K-Nearest Neighbors", f"{knn_acc:.2%}")

# ==================== BAGGING TAB ====================
with tabs[2]:
    st.markdown("## 🎒 Bagging (Bootstrap Aggregating)")

    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #1976d2; margin-top: 0;'>How Bagging Works</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Bagging trains multiple versions of the same algorithm on different random subsets of the training data.
    Each subset is created by randomly sampling with replacement (bootstrap sampling).
    </p>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    <strong>Random Forest</strong> is the most popular bagging method. It creates multiple decision trees,
    each trained on a different random sample of the data.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='success-box'>
    <h4 style='margin-top: 0;'>✨ Key Benefits</h4>
    <ul style='font-size: 1.05rem; line-height: 1.8;'>
        <li>Reduces variance (overfitting)</li>
        <li>Works well with unstable models like decision trees</li>
        <li>Can be trained in parallel for speed</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💻 View Complete Implementation Code"):
        st.markdown("**Option 1: Bagging Classifier**")
        st.code("""
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Create a bagging classifier with decision trees
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,  # Number of trees to create
    random_state=42
)

# Train the ensemble
bagging_clf.fit(X_train, y_train)

# Make predictions
predictions = bagging_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Bagging Classifier Accuracy: {accuracy:.2%}")
        """, language="python")

        st.markdown("**Option 2: Random Forest (Optimized Bagging)**")
        st.code("""
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Random Forest is a specialized bagging method
# It also randomly selects features at each split
rf_clf = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    random_state=42
)

# Train the forest
rf_clf.fit(X_train, y_train)

# Make predictions
predictions = rf_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy:.2%}")

# Get feature importance (which features matter most)
importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print("\\nTop 10 Important Features:")
print(feature_importance_df.head(10))
        """, language="python")

    st.markdown("---")

    n_estimators = st.slider("Number of Models in Ensemble:", 10, 200, 100, 10)

    # Initialize session state for bagging results
    if 'bagging_results' not in st.session_state:
        st.session_state.bagging_results = {}
    if 'rf_results' not in st.session_state:
        st.session_state.rf_results = {}

    col1, col2, col3 = st.columns([1, 1, 0.5])

    with col1:
        if st.button("Run Bagging Classifier", type="primary", key="bagging_btn"):
            with st.spinner("Training bagging ensemble..."):
                # Create a bagging classifier with decision trees
                bagging_clf = BaggingClassifier(
                    estimator=DecisionTreeClassifier(),
                    n_estimators=n_estimators,  # Number of trees
                    random_state=42
                )

                # Train the ensemble
                bagging_clf.fit(X_train, y_train)
                predictions = bagging_clf.predict(X_test)

                # Store results in session state
                st.session_state.bagging_results = {
                    'predictions': predictions,
                    'n_estimators': n_estimators
                }

    with col2:
        if st.button("Run Random Forest", type="primary", key="rf_btn"):
            with st.spinner("Training random forest..."):
                # Random Forest is an optimized bagging method
                rf_clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=10,
                    random_state=42
                )

                # Train the ensemble
                rf_clf.fit(X_train, y_train)
                predictions = rf_clf.predict(X_test)

                # Store results in session state
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_clf.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                st.session_state.rf_results = {
                    'predictions': predictions,
                    'importance_df': importance_df,
                    'n_estimators': n_estimators
                }

    with col3:
        if st.button("Clear Results", key="clear_bagging"):
            st.session_state.bagging_results = {}
            st.session_state.rf_results = {}
            st.rerun()

    # Display results side by side if they exist
    if st.session_state.bagging_results or st.session_state.rf_results:
        st.markdown("---")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if st.session_state.bagging_results:
                st.success("✓ Bagging Classifier - Training Complete!")
                display_results(y_test, st.session_state.bagging_results['predictions'], "Bagging Results")

        with result_col2:
            if st.session_state.rf_results:
                st.success("✓ Random Forest - Training Complete!")
                display_results(y_test, st.session_state.rf_results['predictions'], "Random Forest Results")

                # Show feature importance
                st.subheader("Top 10 Most Important Features")
                importance_df = st.session_state.rf_results['importance_df']

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Importance')
                ax.invert_yaxis()
                st.pyplot(fig)

# ==================== BOOSTING TAB ====================
with tabs[3]:
    st.markdown("## 🚀 Boosting")

    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #1976d2; margin-top: 0;'>How Boosting Works</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Boosting trains models sequentially, where each new model focuses on correcting
    the errors made by previous models. Unlike bagging, models are dependent on each other.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(90deg, #f3e5f5 0%, #ffffff 100%); padding: 15px; border-radius: 8px; margin: 15px 0;'>
    <h4 style='margin-top: 0; color: #6a1b9a;'>Popular Boosting Algorithms</h4>
    <ul style='font-size: 1.05rem; line-height: 1.8;'>
        <li><strong>AdaBoost:</strong> Adjusts weights of misclassified samples</li>
        <li><strong>Gradient Boosting:</strong> Builds trees to predict residual errors</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💻 View Complete Implementation Code"):
        st.markdown("**Option 1: AdaBoost (Adaptive Boosting)**")
        st.code("""
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# AdaBoost increases weight on misclassified samples
# Each tree focuses on examples previous trees got wrong
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,      # Number of boosting rounds
    learning_rate=1.0,    # Step size for weight updates
    random_state=42
)

# Train sequentially - each model learns from previous mistakes
ada_clf.fit(X_train, y_train)

# Make predictions
predictions = ada_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"AdaBoost Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))
        """, language="python")

        st.markdown("**Option 2: Gradient Boosting**")
        st.code("""
# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Gradient Boosting builds trees on prediction errors (residuals)
# Each new tree tries to fix the errors of all previous trees
gb_clf = GradientBoostingClassifier(
    n_estimators=50,      # Number of trees
    learning_rate=0.1,    # How much each tree contributes
    max_depth=3,          # Depth of each tree
    random_state=42
)

# Train sequentially - each tree predicts residuals
gb_clf.fit(X_train, y_train)

# Make predictions
predictions = gb_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Gradient Boosting Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature importance
print("\\nTop 10 Important Features:")
importances = gb_clf.feature_importances_
for i, imp in enumerate(importances[:10]):
    print(f"Feature {i}: {imp:.4f}")
        """, language="python")

    st.markdown("---")

    boost_n_estimators = st.slider("Number of Boosting Rounds:", 10, 200, 50, 10)

    # Initialize session state for boosting results
    if 'ada_results' not in st.session_state:
        st.session_state.ada_results = {}
    if 'gb_results' not in st.session_state:
        st.session_state.gb_results = {}

    col1, col2, col3 = st.columns([1, 1, 0.5])

    with col1:
        if st.button("Run AdaBoost", type="primary", key="ada_btn"):
            with st.spinner("Training AdaBoost..."):
                # AdaBoost increases weight on misclassified samples
                ada_clf = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=boost_n_estimators,
                    learning_rate=1.0,
                    random_state=42
                )

                ada_clf.fit(X_train, y_train)
                predictions = ada_clf.predict(X_test)

                # Store results in session state
                st.session_state.ada_results = {
                    'predictions': predictions,
                    'n_estimators': boost_n_estimators
                }

    with col2:
        if st.button("Run Gradient Boosting", type="primary", key="gb_btn"):
            with st.spinner("Training Gradient Boosting..."):
                # Gradient Boosting builds trees on prediction residuals
                gb_clf = GradientBoostingClassifier(
                    n_estimators=boost_n_estimators,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )

                gb_clf.fit(X_train, y_train)
                predictions = gb_clf.predict(X_test)

                # Store results in session state
                st.session_state.gb_results = {
                    'predictions': predictions,
                    'n_estimators': boost_n_estimators
                }

    with col3:
        if st.button("Clear Results", key="clear_boosting"):
            st.session_state.ada_results = {}
            st.session_state.gb_results = {}
            st.rerun()

    # Display results side by side if they exist
    if st.session_state.ada_results or st.session_state.gb_results:
        st.markdown("---")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if st.session_state.ada_results:
                st.success("✓ AdaBoost - Training Complete!")
                display_results(y_test, st.session_state.ada_results['predictions'], "AdaBoost Results")

        with result_col2:
            if st.session_state.gb_results:
                st.success("✓ Gradient Boosting - Training Complete!")
                display_results(y_test, st.session_state.gb_results['predictions'], "Gradient Boosting Results")

# ==================== STACKING TAB ====================
with tabs[4]:
    st.markdown("## 📚 Stacking")

    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #1976d2; margin-top: 0;'>How Stacking Works</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Stacking uses multiple different models (base learners) and then trains a final model
    (meta-learner) on their predictions. The meta-learner learns how to best combine the base model predictions.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='success-box'>
    <h4 style='margin-top: 0;'>🔄 Stacking Process</h4>
    <ol style='font-size: 1.05rem; line-height: 1.8;'>
        <li>Train several base models on the training data</li>
        <li>Use these models to make predictions</li>
        <li>Train a final model using the base model predictions as features</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Add stacking diagram
    st.markdown("## 🏗️ Stacking Architecture")
    try:
        st.image("stacking.png", width=600,
                caption="Stacking: Base models trained with cross-validation, meta-model combines predictions")
    except:
        st.info("Stacking diagram not found")

    st.markdown("---")

    with st.expander("💻 View Complete Implementation Code"):
        st.markdown("**Complete Stacking Implementation**")
        st.code("""
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Define Base Learners (Level 0)
# These are different types of models that make initial predictions
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
]

# Step 2: Define Meta-Learner (Level 1)
# This model learns how to best combine the base model predictions
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# Step 3: Create Stacking Classifier
# StackingClassifier automatically handles cross-validation
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5  # Use 5-fold cross-validation to avoid overfitting
)

# Step 4: Train the Stacking Ensemble
# This automatically:
# 1. Trains each base learner on different folds
# 2. Generates out-of-fold predictions
# 3. Trains meta-learner on these predictions
stacking_clf.fit(X_train, y_train)

# Step 5: Make Final Predictions
predictions = stacking_clf.predict(X_test)

# Step 6: Evaluate Performance
accuracy = accuracy_score(y_test, predictions)
print(f"Stacking Classifier Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))

# Compare with individual base learners
print("\\nBase Learner Performance:")
for name, model in base_learners:
    model.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {base_acc:.2%}")
        """, language="python")

    st.markdown("---")

    if st.button("Run Stacking Classifier", type="primary"):
        with st.spinner("Training stacking ensemble..."):
            # Define base learners (level 0)
            base_learners = [
                ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
            ]

            # Define meta-learner (level 1)
            # This model learns to combine predictions from base learners
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)

            # Create stacking classifier
            stacking_clf = StackingClassifier(
                estimators=base_learners,
                final_estimator=meta_learner,
                cv=5  # Use 5-fold cross-validation to generate predictions
            )

            # Train the stacking ensemble
            stacking_clf.fit(X_train, y_train)
            predictions = stacking_clf.predict(X_test)

            st.success("Training complete!")
            display_results(y_test, predictions, "Stacking Results")

            # Compare with base learners
            st.subheader("Base Learner Performance")
            col1, col2, col3 = st.columns(3)

            with col1:
                dt = DecisionTreeClassifier(max_depth=5, random_state=42)
                dt.fit(X_train, y_train)
                dt_acc = accuracy_score(y_test, dt.predict(X_test))
                st.metric("Decision Tree", f"{dt_acc:.2%}")

            with col2:
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                knn_acc = accuracy_score(y_test, knn.predict(X_test))
                st.metric("K-Nearest Neighbors", f"{knn_acc:.2%}")

            with col3:
                rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf.fit(X_train, y_train)
                rf_acc = accuracy_score(y_test, rf.predict(X_test))
                st.metric("Random Forest", f"{rf_acc:.2%}")

# ==================== BLENDING TAB ====================
with tabs[5]:
    st.markdown("## 🎨 Blending")

    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #1976d2; margin-top: 0;'>How Blending Works</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Blending is similar to stacking but simpler. Instead of using cross-validation,
    we split the training data into two parts. Blending is faster than stacking but may not use data as efficiently.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
    <h4 style='margin-top: 0;'>📋 Data Split Strategy</h4>
    <ul style='font-size: 1.05rem; line-height: 1.8;'>
        <li><strong>Train Set:</strong> Used to train base models</li>
        <li><strong>Validation Set:</strong> Used to create predictions for training the meta-model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💻 View Complete Implementation Code"):
        st.markdown("**Complete Blending Implementation**")
        st.code("""
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Split Training Data for Blending
# 70% for training base models, 30% for training meta-model
X_base_train, X_blend_train, y_base_train, y_blend_train = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Step 2: Train Base Models on Base Training Set
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

dt.fit(X_base_train, y_base_train)
knn.fit(X_base_train, y_base_train)
rf.fit(X_base_train, y_base_train)

# Step 3: Generate Predictions on Blend Training Set
# These predictions become features for the meta-model
dt_blend_pred = dt.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)
knn_blend_pred = knn.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)
rf_blend_pred = rf.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)

# Combine predictions into a new feature matrix
X_blend = np.hstack([dt_blend_pred, knn_blend_pred, rf_blend_pred])

# Step 4: Train Meta-Model on Blended Features
# Meta-model learns optimal weights for base model predictions
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_blend, y_blend_train)

# Step 5: Make Final Predictions on Test Set
# First, get base model predictions on test set
dt_test_pred = dt.predict_proba(X_test)[:, 1].reshape(-1, 1)
knn_test_pred = knn.predict_proba(X_test)[:, 1].reshape(-1, 1)
rf_test_pred = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Combine test predictions
X_test_blend = np.hstack([dt_test_pred, knn_test_pred, rf_test_pred])

# Final prediction from meta-model
predictions = meta_model.predict(X_test_blend)

# Step 6: Evaluate Performance
accuracy = accuracy_score(y_test, predictions)
print(f"Blending Accuracy: {accuracy:.2%}")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))

# Show meta-model weights (how much it trusts each base model)
print("\\nMeta-Model Learned Weights:")
print(f"Decision Tree weight: {meta_model.coef_[0][0]:.4f}")
print(f"KNN weight: {meta_model.coef_[0][1]:.4f}")
print(f"Random Forest weight: {meta_model.coef_[0][2]:.4f}")
        """, language="python")

    st.markdown("---")

    if st.button("Run Blending", type="primary"):
        with st.spinner("Training blending ensemble..."):
            # Split training data for blending
            # 70% for training base models, 30% for training meta-model
            X_base_train, X_blend_train, y_base_train, y_blend_train = train_test_split(
                X_train, y_train, test_size=0.3, random_state=42
            )

            # Step 1: Train base models on base training set
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=5)
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

            dt.fit(X_base_train, y_base_train)
            knn.fit(X_base_train, y_base_train)
            rf.fit(X_base_train, y_base_train)

            # Step 2: Generate predictions on blend training set
            # These predictions become features for the meta-model
            dt_blend_pred = dt.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)
            knn_blend_pred = knn.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)
            rf_blend_pred = rf.predict_proba(X_blend_train)[:, 1].reshape(-1, 1)

            # Combine predictions into a new feature matrix
            X_blend = np.hstack([dt_blend_pred, knn_blend_pred, rf_blend_pred])

            # Step 3: Train meta-model on blended predictions
            meta_model = LogisticRegression(random_state=42)
            meta_model.fit(X_blend, y_blend_train)

            # Step 4: Make final predictions on test set
            # First get base model predictions on test set
            dt_test_pred = dt.predict_proba(X_test)[:, 1].reshape(-1, 1)
            knn_test_pred = knn.predict_proba(X_test)[:, 1].reshape(-1, 1)
            rf_test_pred = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)

            # Combine test predictions
            X_test_blend = np.hstack([dt_test_pred, knn_test_pred, rf_test_pred])

            # Final prediction from meta-model
            predictions = meta_model.predict(X_test_blend)

            st.success("Training complete!")
            display_results(y_test, predictions, "Blending Results")

            # Show meta-model weights
            st.subheader("Meta-Model Learned Weights")
            st.write("These weights show how much the meta-model trusts each base model:")

            weights_df = pd.DataFrame({
                'Base Model': ['Decision Tree', 'K-Nearest Neighbors', 'Random Forest'],
                'Weight': meta_model.coef_[0]
            })

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(weights_df['Base Model'], weights_df['Weight'])
            ax.set_xlabel('Weight')
            st.pyplot(fig)

            st.write("""
            **Interpreting Weights:**
            - Positive weights mean the model's predictions are valuable
            - Larger absolute values mean more influence on final prediction
            - The meta-model learned these weights automatically
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
    <h4 style='color: #1f77b4; margin-bottom: 10px;'>SEAS-8505: Applied Machine Intelligence and Reinforcement Learning</h4>
    <p style='color: #555; font-size: 1.1rem;'><strong>Created by: Dr. Elbasheer</strong></p>
    <p style='color: #666; margin-top: 10px;'>Experiment with different techniques to discover which works best for your data.</p>
</div>
""", unsafe_allow_html=True)

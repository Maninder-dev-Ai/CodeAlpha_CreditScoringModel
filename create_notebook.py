import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Introduction
cells.append(nbf.v4.new_markdown_cell("""# Credit Scoring Model
This notebook builds a complete Machine Learning pipeline to predict whether a customer is creditworthy using the German Credit Data dataset from UCI.

**Pipeline Steps:**
1. Import dataset
2. Exploratory Data Analysis (EDA)
3. Handle missing values
4. Remove duplicates
5. Encode categorical features
6. Feature Engineering
7. Feature Scaling
8. Train/Test Split
9. Train multiple models
10. Compare model performances
11. Evaluate the best model
12. Save trained model
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')
"""))

# Cell 3: Fetch Data
cells.append(nbf.v4.new_markdown_cell("""## 1. Import Dataset
We will download the German Credit dataset directly.
"""))

cells.append(nbf.v4.new_code_cell("""# Download the dataset using pandas
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
# The dataset has 20 features and 1 target variable
columns = ['status_checking_acc', 'duration_months', 'credit_history', 'purpose', 
           'credit_amount', 'savings_acc', 'employment_st', 'installment_rate', 
           'personal_status', 'other_debtors', 'residence_since', 'property', 
           'age_years', 'other_installment_plans', 'housing', 'existing_credits', 
           'job', 'num_dependents', 'telephone', 'foreign_worker', 'target']

df = pd.read_csv(url, sep=' ', header=None, names=columns)
df.to_csv('dataset/german_credit.csv', index=False)
print("Dataset shape:", df.shape)
df.head()
"""))

# Cell 4: EDA
cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis
Let's explore the data.
"""))

cells.append(nbf.v4.new_code_cell("""# Check target distribution
# Target mapping: 1 = Good, 2 = Bad -> We will map to 1 = Good, 0 = Bad
df['target'] = df['target'].map({1: 1, 2: 0})

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable (0 = Bad, 1 = Good)')
plt.savefig('screenshots/target_distribution.png')
plt.show()
"""))

# Cell 5: Missing values & Duplicates
cells.append(nbf.v4.new_markdown_cell("""## 3. & 4. Handle Missing Values and Duplicates
"""))

cells.append(nbf.v4.new_code_cell("""# Check missing values
print("Missing values in each column:\\n", df.isnull().sum())

# Drop duplicates if any
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df.drop_duplicates(inplace=True)
"""))

# Cell 6: Encoding
cells.append(nbf.v4.new_markdown_cell("""## 5. Encode Categorical Features
"""))

cells.append(nbf.v4.new_code_cell("""categorical_cols = df.select_dtypes(include=['object']).columns

# We will use LabelEncoder for simplicity and standard processing
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical columns encoded.")
df.head()
"""))

# Cell 7: Feature Engineering & Scaling
cells.append(nbf.v4.new_markdown_cell("""## 6. & 7. Feature Engineering and Scaling
"""))

cells.append(nbf.v4.new_code_cell("""# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('screenshots/correlation_heatmap.png')
plt.show()

# Split X and y
X = df.drop('target', axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
"""))

# Cell 8: Train/Test Split
cells.append(nbf.v4.new_markdown_cell("""## 8. Split Dataset into Train/Test
"""))

cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
"""))

# Cell 9: Train Models
cells.append(nbf.v4.new_markdown_cell("""## 9. Train Multiple Models
We will train Logistic Regression, Decision Tree, and Random Forest.
"""))

cells.append(nbf.v4.new_code_cell("""# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

results_df = pd.DataFrame(results)
"""))

# Cell 10: Compare
cells.append(nbf.v4.new_markdown_cell("""## 10. & 11. Compare Performances & Select Best
"""))

cells.append(nbf.v4.new_code_cell("""print("Model Performance Comparison:")
display(results_df)

# Select best model based on F1 Score or ROC-AUC
best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax()]['Model']
best_model = models[best_model_name]
print(f"\\nBest Model selected: {best_model_name}")
"""))

# Cell 11: Evaluate Best Model
cells.append(nbf.v4.new_markdown_cell("""## 12. & 13. Evaluate Best Model & Plot Graphs
"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate best model
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('screenshots/confusion_matrix.png')
plt.show()

# Feature Importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('screenshots/feature_importance.png')
    plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.savefig('screenshots/roc_curve.png')
plt.show()
"""))

# Cell 12: Save Model
cells.append(nbf.v4.new_markdown_cell("""## 14. Save Trained Model
"""))

cells.append(nbf.v4.new_code_cell("""joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Save label encoders to use them later
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and preprocessors saved successfully.")
"""))

# Cell 13: Prediction Example
cells.append(nbf.v4.new_markdown_cell("""## 15. Prediction Example
"""))

cells.append(nbf.v4.new_code_cell("""# Example prediction
def predict_creditworthiness(customer_data):
    # Load models
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Scale data
    scaled_data = scaler.transform([customer_data])
    
    # Predict
    pred = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[:, 1]
    
    return "Creditworthy (Good)" if pred[0] == 1 else "Not Creditworthy (Bad)", prob[0]

# Let's take the first row of X_test (before scaling) to test
# Since we lost unscaled X_test, we'll take the first row of X
example_customer = X.iloc[0].values
status, probability = predict_creditworthiness(example_customer)
print(f"Prediction: {status}")
print(f"Probability of being creditworthy: {probability:.2f}")
"""))

nb.cells = cells

with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")

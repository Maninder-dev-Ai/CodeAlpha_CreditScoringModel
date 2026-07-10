# Credit Scoring Model

## Project Overview
This project develops a complete Machine Learning pipeline to predict whether a customer is creditworthy based on their historical financial data. The model is built using Python, Scikit-Learn, and Pandas. It's intended to be a production-quality implementation suitable for deployment.

## Dataset
The project uses the **German Credit Data** from the UCI Machine Learning Repository.
It contains 1,000 instances with 20 categorical and numerical attributes, detailing various aspects of a customer's credit history, employment, and personal status, alongside a target variable indicating credit risk (Good/Bad).

## ML Pipeline
1. **Import dataset**: Data is automatically downloaded and loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis**: Visualizations of target distributions and feature correlations.
3. **Data Preprocessing**: Handling missing values and removing duplicates.
4. **Encoding Categorical Features**: Using Label Encoding for categorical strings.
5. **Feature Engineering & Scaling**: Standardization of numerical features using `StandardScaler`.
6. **Train/Test Split**: 80/20 stratified split.
7. **Model Training**: Evaluation of multiple algorithms.
8. **Model Selection & Evaluation**: Identifying the best model and assessing its performance with comprehensive metrics.
9. **Export**: Saving the model and preprocessors for inference.

## Algorithms Used
We evaluated the following models:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

## Results & Accuracy
The **Random Forest** and **Logistic Regression** models were compared. Based on ROC-AUC and F1 Score metrics on the test set, the optimal model is selected programmatically in the notebook. Key evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Visualizations such as Correlation Heatmaps, Feature Importance charts, ROC Curves, and Confusion Matrices are generated and saved in the `screenshots/` folder.

## Installation Steps
1. Clone or download this repository.
2. Ensure you have Python installed (e.g., Python 3.8+).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Navigate to the project directory.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Alternatively, you can run all cells automatically (if `nbconvert` is installed):
   ```bash
   python -m nbconvert --execute notebook.ipynb
   ```
4. The notebook will generate the datasets, trained model (`model.pkl`), scalers, and all visual screenshots in the respective directories.

## Future Improvements
- **Hyperparameter Tuning**: Implementing GridSearchCV or RandomizedSearchCV to further optimize the chosen model.
- **Handling Imbalance**: Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
- **Advanced Feature Engineering**: Creating more complex derived features to capture non-linear relationships.
- **Deployment**: Wrapping the model in a Flask or FastAPI application for real-time predictions via an API.

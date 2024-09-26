# Titanic - Machine Learning from Disaster

This project uses machine learning techniques to predict the survival of passengers on the Titanic. We used the famous Titanic dataset from Kaggle, which includes information such as passenger class, age, sex, fare, and other features to predict whether a passenger survived or not.

## Project Overview

We implemented two versions of a machine learning model:
1. `model.py`: A stacking model using Random Forest, XGBoost, and Logistic Regression.
2. `model_v2.py`: An improved version of the stacking model that also includes LightGBM and additional feature engineering.

### Data Preprocessing

1. **Feature Engineering**: 
   - `FamilySize`: Combined number of siblings/spouses and parents/children aboard the Titanic.
   - `IsAlone`: Binary variable indicating whether the passenger is alone.
   - `FarePerPerson`: Fare divided by `FamilySize`.
   - `AgeBin`: Age categorized into bins.
   - `Deck`: Extracted from the `Cabin` field.

2. **Imputation**:
   - `Age`, `Fare`, and `FarePerPerson` were imputed using the median of the respective columns.
   - `AgeBin` was filled with the most frequent bin.
   
3. **One-Hot Encoding**: 
   - Categorical variables such as `Sex`, `Embarked`, and `Deck` were one-hot encoded.

4. **Scaling**: 
   - Applied standard scaling to the features before feeding them into the model.

## Models

### model.py

- Base learners: RandomForest, XGBoost, and Logistic Regression.
- Final model: Logistic Regression.
- Cross-validation: StratifiedKFold with 5 splits.
- Performance on Kaggle: **0.77751** (135 survivors predicted).

### model_v2.py

- Base learners: RandomForest, XGBoost, LightGBM, and Logistic Regression.
- Final model: Gradient Boosting.
- Cross-validation: StratifiedKFold with 5 splits.
- Performance on Kaggle: **0.75598** (128 survivors predicted).

### Model Comparison

| Model            | Public Score | Predicted Survivors |
|------------------|--------------|---------------------|
| Stacking (model) | 0.77751      | 135                 |
| Stacking (model_v2) | 0.75598   | 128                 |

### Files
`train.csv` - Training dataset containing passenger data and survival outcomes.
`test.csv` - Test dataset containing passenger data without survival outcomes.

Two versions of the model are provided:
1. **model.py**: A stacking model that uses RandomForest, XGBoost, and Logistic Regression. This model performed better, achieving a public score of **0.77751** on Kaggle.
2. **model_v2.py**: An improved version that incorporates LightGBM and Gradient Boosting as the final estimator. However, it achieved a slightly lower score of **0.75598**.

### Key Features:
- **Feature Engineering**: Added important features such as `FamilySize`, `IsAlone`, `FarePerPerson`, and `AgeBin`, which helped improve the model's performance.
- **Stacking Models**: We used a combination of RandomForest, XGBoost, LightGBM, and Logistic Regression in a stacking classifier.
- **Cross-Validation**: The model used stratified k-fold cross-validation to ensure robust performance.

The repository includes two Python scripts (`model.py` and `model_v2.py`), the preprocessed datasets, and the submission files for Kaggle.

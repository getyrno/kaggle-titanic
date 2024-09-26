import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

train_data['Deck'] = train_data['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'M')
test_data['Deck'] = test_data['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'M')

train_data = pd.get_dummies(train_data, columns=['Deck'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Deck'], drop_first=True)

missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

test_data = test_data[train_data.columns.drop('Survived')]

train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

fare_imputer = SimpleImputer(strategy='median')
test_data['Fare'] = fare_imputer.fit_transform(test_data[['Fare']])

age_imputer = SimpleImputer(strategy='median')
train_data['Age'] = age_imputer.fit_transform(train_data[['Age']])
test_data['Age'] = age_imputer.transform(test_data[['Age']])

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels=[1, 2, 3, 4]).astype(int)
test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels=[1, 2, 3, 4]).astype(int)

X = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
y = train_data['Survived']
X_test = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000))
]

stack_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(), cv=5)

skf = StratifiedKFold(n_splits=5)

cv_scores = cross_val_score(stack_model, X, y, cv=skf)

stack_model.fit(X, y)

predictions = stack_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('stacking_submission.csv', index=False)
# kaggle 0.77751 (Survived: 135)
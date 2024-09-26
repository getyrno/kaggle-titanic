import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

train_data['FarePerPerson'] = train_data['Fare'] / train_data['FamilySize']
test_data['FarePerPerson'] = test_data['Fare'] / test_data['FamilySize']

train_data['AgeBin'] = pd.cut(train_data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=[1, 2, 3, 4, 5])
test_data['AgeBin'] = pd.cut(test_data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=[1, 2, 3, 4, 5])

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

imputer = SimpleImputer(strategy='median')
train_data[['Age', 'Fare', 'FarePerPerson']] = imputer.fit_transform(train_data[['Age', 'Fare', 'FarePerPerson']])
test_data[['Age', 'Fare', 'FarePerPerson']] = imputer.transform(test_data[['Age', 'Fare', 'FarePerPerson']])

train_data['AgeBin'] = train_data['AgeBin'].fillna(train_data['AgeBin'].mode()[0])
test_data['AgeBin'] = test_data['AgeBin'].fillna(test_data['AgeBin'].mode()[0])

train_data['AgeBin'] = train_data['AgeBin'].astype(float)
test_data['AgeBin'] = test_data['AgeBin'].astype(float)

X = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
y = train_data['Survived']
X_test = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

if np.any(np.isnan(X)) or np.any(np.isnan(X_test)):
    print("found NaN in data")

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000))
]

stack_model = StackingClassifier(estimators=base_learners, final_estimator=GradientBoostingClassifier(), cv=5)

skf = StratifiedKFold(n_splits=5)

cv_scores = cross_val_score(stack_model, X, y, cv=skf)

stack_model.fit(X, y)

predictions = stack_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('improved_stacking_submission.csv', index=False)
# kaggle 0.75598 (Survived: 128)
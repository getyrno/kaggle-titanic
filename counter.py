import pandas as pd

# Загрузим тестовый файл и предсказанные данные
predictions = pd.read_csv('gender_submission.csv')

# Посчитаем количество предсказанных выживших
survived_count = predictions['Survived'].sum()
print(survived_count)

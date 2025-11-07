from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import m2cgen as m2c

df = pd.read_csv('prepared_dataset.csv')

# Разбиение на обучающую и тестовую выборки
x = df.drop(columns=['Label'])
y = df['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Обучение модели алгоритмом Random Forest
rf = RandomForestClassifier(
    n_estimators=7,        
    max_depth=8,
    max_features='sqrt',
    random_state=42,
    n_jobs=1               
)

rf.fit(x_train, y_train)

# Генерация C кода
c_code = m2c.export_to_c(rf, function_name="predict_risk")

# Сохранение в файл
with open("random_forest_model.c", "w") as f:
    f.write(c_code)

print("C код успешно сгенерирован!")

# Создание заголовочного файла
header_code = """
#ifndef RANDOM_FOREST_MODEL_H
#define RANDOM_FOREST_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

	void predict_risk(double* input, double* output);

#ifdef __cplusplus
}
#endif

#endif
"""

with open("random_forest_model.h", "w") as f:
    f.write(header_code)


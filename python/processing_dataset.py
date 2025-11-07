import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('df_balanced.csv')

# Удаление информационных данных, которые не должны учитываться при построении модели
df = df.drop(columns=['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Protocol'])

# Кодирование столбца атак методом LabelEncoder
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print("Классы в порядке кодировки:")
print(le.classes_)

# Удаление строк с пустыми значениями или inf
df = df.apply(pd.to_numeric, errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# Поиск сильной зависимости целевой переменной от какого-то из показателей алгоритмом Random Forest
X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns)
top = importances.sort_values(ascending=False).head(10)
print("Топ-10 важнейших признаков:")
print(top)

# Поиск пар показателей с высоким коэффициентов корреляции
corr_matrix = X.corr()
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]
print("Самые сильные корреляции:")
print(corr_pairs.head(10))

# Удаление признаков, которые в имели высокий коэффициент корреляции с другими признаками
df = df.drop(columns = ['Fwd Header Length', 'Bwd Header Length', 'Fwd IAT Total', 'Packet Length Mean',
                        'Flow IAT Max', 'Bwd Packet Length Max', 'Fwd Packets/s', 'Packet Length Std', 'Idle Max',
                        'Bwd Packet Length Std', 'Idle Min', 'Fwd Packet Length Std', 'Subflow Bwd Packets', 'Total Backward Packets',
                        'Idle Mean', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean', 'Flow IAT Std', 'Fwd IAT Std', 'Fwd Header Length.1'])

# Проверка коэффициентов корреляции после удаления
X = df.drop(columns=['Label'])
y = df['Label']
corr_matrix = X.corr()
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]
print("Самые сильные корреляции после удаления сильнокоррелирующих признаков:")
print(corr_pairs.head(10))

df.to_csv('prepared_dataset.csv', index=False)

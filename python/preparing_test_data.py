import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('test_data.csv')

# Удаление информационных столбцов
df.columns = df.columns.str.strip()
df.loc[df['Label'].isin([
    'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'
]), 'Label'] = 'DoS'
df = df.drop(columns=['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Protocol'])

# Кодирование столбца целевой переменной
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Удаление строк с пустыми значениями или inf
df = df.apply(pd.to_numeric, errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# Удаление столбцов с сильной корреляцией
df = df.drop(columns = ['Fwd Header Length', 'Bwd Header Length', 'Fwd IAT Total', 'Packet Length Mean',
                        'Flow IAT Max', 'Bwd Packet Length Max', 'Fwd Packets/s', 'Packet Length Std', 'Idle Max',
                        'Bwd Packet Length Std', 'Idle Min', 'Fwd Packet Length Std', 'Subflow Bwd Packets', 'Total Backward Packets',
                        'Idle Mean', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean', 'Flow IAT Std', 'Fwd IAT Std', 'Fwd Header Length.1'])

# Сохранение данных с истинными значениями целевой переменной
df.to_csv('data_with_label.csv', index=False, header=False)

# Сохранение данных без значений целевой переменной
df1 = df.drop(columns = ['Label'])
df1.to_csv('data_without_label.csv', index=False, header=False)
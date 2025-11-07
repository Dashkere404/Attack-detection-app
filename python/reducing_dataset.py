import pandas as pd
import numpy as np

df = pd.read_csv('combined_dataset.csv', dtype_backend='pyarrow')
# Удаление лишних пробелов в столбцах
df.columns = df.columns.str.strip()


# Удаление всех строк, где есть хотя бы одно пустое значение, а также удаление строк с бесконечными значениями
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# Объединение всех Dos атак в один класс - Dos
df.loc[df['Label'].isin([
    'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'
]), 'Label'] = 'DoS'

# Оставление строк с нужными параметрами атак
classes = ['BENIGN', 'DoS', 'DDoS', 'PortScan']
df = df[df['Label'].isin(classes)]

# Сокращение каждого класса атак до 30000 экземпляров (если меньше, то оставляем столько, сколько есть)
balanced_dfs = []
for cls in classes:
    cls_data = df[df['Label'] == cls]
    if len(cls_data) > 30000:
        cls_data = cls_data.sample(n=30000, random_state=42)
    balanced_dfs.append(cls_data)
df = pd.concat(balanced_dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Распределение после балансировки:")
print(df['Label'].value_counts())

df.to_csv("df_balanced.csv")

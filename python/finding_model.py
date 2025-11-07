import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv('prepared_dataset.csv')

# Разделение на обучающую и тестовую выборки
x = df.drop(columns=['Label'])
y = df['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Функция вычисления метрик качества модели
def find_metrics(y_true, y_pred, y_pred_time):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Prediction time': y_pred_time
    }
    return metrics

# Функция обучения моделей
def evaluate_model(model, x_train, x_test, y_train, y_test, model_name, needs_scaling=False, random_state=42):
    if needs_scaling:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model.fit(x_train_scaled, y_train)
        start = time.time()
        y_pred = model.predict(x_test_scaled)
    else:
        model.fit(x_train, y_train)
        start = time.time()
        y_pred = model.predict(x_test)

    pred_time_total = time.time() - start
    my_metrics = find_metrics(y_test, y_pred, pred_time_total)
    my_metrics['Model'] = model_name
    return my_metrics

results = []

# CART
cart = DecisionTreeClassifier(max_depth=8, random_state=42, criterion='gini')
results.append(evaluate_model(cart, x_train, x_test, y_train, y_test, "CART"))

# Random Forest
rf = RandomForestClassifier(n_estimators=7, max_depth=8, max_features='sqrt', random_state=42, n_jobs=1)
results.append(evaluate_model(rf, x_train, x_test, y_train, y_test, "Random Forest"))

# Logistic Regression (с масштабированием)
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
results.append(evaluate_model(lr, x_train, x_test, y_train, y_test, "Logistic Regression", needs_scaling=True))

# Naive Bayes
nb = GaussianNB()
results.append(evaluate_model(nb, x_train, x_test, y_train, y_test, "Naive Bayes"))

# LDA
lda = LinearDiscriminantAnalysis()
results.append(evaluate_model(lda, x_train, x_test, y_train, y_test, "LDA"))

# Вывод результатов вычисления метрик
df_results = pd.DataFrame(results)
df_results.set_index('Model', inplace=True)
print(df_results)
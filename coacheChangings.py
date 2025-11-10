import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("data/cleaned/coaches_cleaned.csv")
    print(f"Dados dos treinadores carregados com sucesso. Dimensão: {df.shape}")
except FileNotFoundError:
    print("❌ ERRO: O ficheiro não foi encontrado")
    exit()

df.sort_values(['tmID', 'year'], inplace=True)

df['next_coachID'] = df.groupby('tmID')['coachID'].shift(-1)

df.dropna(subset=['next_coachID'], inplace=True)

df['coach_change'] = (df['coachID'] != df['next_coachID']).astype(int)

print("Variável alvo 'coach_change criada")
print("Distribuiçáo da variável alvo:")
print(df['coach_change'].value_counts(normalize=True))

features = [
    'win_pct',
    'post_win_pct',
    'tenure_years',
    'performance_trend',
    'won',
    'lost'
]
target = 'coach_change'

df['performance_trend'] = df['performance_trend'].fillna(0)
df['post_win_pct'] = df['post_win_pct'].fillna(0)

df.dropna(subset=features, inplace=True)

X = df[features]
y = df[target]

print(f"Features selecionadas: {', '.join(features)}")
print(f"Dimensão final do dataset para o modelo: {X.shape}")

print("\n Dividindo os dados (treino e teste cronológico)...")

latest_season = df['year'].max()
train_df = df[df['year'] < latest_season]
test_df = df[df['year'] == latest_season]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Treino: {len(X_train)} amostras (épocas até {int(latest_season-1)})")
print(f"Teste: {len(X_test)} amostras (época {int(latest_season)})")

print("\n Treinando o modelo de classificação...")

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=5)
model.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

print("\n Avaliando o desempenho do modelo...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_log_proba(X_test)[:, 1]

print("\n--- Matriz de Confusão ---")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Mudou', 'Mudou'], yticklabels=['Não mudou', 'Mudou'])
plt.ylabel('Valor Real')
plt.xlabel('Valor previsto')
plt.title('Matriz de confusão')
plt.show()

print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=['Não Mudou (0)', 'Mudou (1)'], labels=[0,1], zero_division=0))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"--- AUC-ROC Score ---")
print(f"AUC-ROC: {auc_score:.4f}\n")

print("\nAnálise de importância das features...")

importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=importances, palette='viridis')
plt.title('Importância das features para prever mudança de treinador')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
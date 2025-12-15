import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# CONFIGURAÇÃO
# =========================================================
PREDICTION_YEAR = 11
LAST_TRAIN_YEAR = 10

# Setup visual
sns.set_style("whitegrid")
output_dir = "results/predictions"
plot_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"========================================================")
print(f"COACH FIRING PREDICTION (HOT SEAT MODEL)")
print(f"========================================================")

# =========================================================
# 1. CARREGAR DADOS
# =========================================================
coaches_path = "data/cleaned/coaches_cleaned.csv"
rankings_path = f"results/predictions/season_{PREDICTION_YEAR}_rankings.csv"

if not os.path.exists(coaches_path):
    raise FileNotFoundError(f"File not found: {coaches_path}")

print("--> Loading Coaches Data...")
df = pd.read_csv(coaches_path)

# Carregar previsões de ranking (do script anterior)
if os.path.exists(rankings_path):
    print("--> Loading Season 11 Predicted Standings...")
    rankings_df = pd.read_csv(rankings_path)
    
    # Criar mapa: {TeamID: Predicted_Win_Pct}
    # Garantir que os IDs são strings
    rankings_df['tmID'] = rankings_df['tmID'].astype(str)
    pred_map = dict(zip(rankings_df.tmID, rankings_df.predicted_win_pct))
    has_predictions = True
else:
    print("WARNING: Rankings file not found. Predictions will be inaccurate (assuming 0 wins).")
    has_predictions = False
    pred_map = {}

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# Garantir ordem correta
df = df.sort_values(['tmID', 'year', 'stint'])

# Filtrar: Queremos apenas o último treinador de cada ano por equipa
# (Geralmente é quem começa o ano seguinte, ou quem é avaliado no final da época)
df_yearly = df.drop_duplicates(subset=['year', 'tmID'], keep='last').copy()

# --- 2.1 INJETAR PREVISÕES NA SEASON 11 ---
# Substituir o win_pct (que é 0.0) pela previsão do rankings.py
if has_predictions:
    mask_s11 = df_yearly['year'] == PREDICTION_YEAR
    # Mapear as previsões
    predicted_values = df_yearly.loc[mask_s11, 'tmID'].map(pred_map)
    # Preencher onde encontrou previsão (se não encontrar, mantém o que está)
    df_yearly.loc[mask_s11, 'win_pct'] = predicted_values.fillna(df_yearly.loc[mask_s11, 'win_pct'])
    print("--> Updated Season 11 win_pct with predicted values.")

# --- 2.2 DEFINIR O TARGET (Mudança de Treinador) ---
# Se o coachID do próximo ano for diferente, houve mudança.
df_yearly['next_coachID'] = df_yearly.groupby('tmID')['coachID'].shift(-1)
df_yearly['coach_change'] = (df_yearly['coachID'] != df_yearly['next_coachID']).astype(int)

# Nota: Na última linha de cada equipa (S11), next_coachID é NaN. 
# Isso não faz mal, pois S11 é o conjunto de TESTE (queremos prever, não treinar).

# --- 2.3 FEATURES AVANÇADAS ---

# Trend: Performance atual vs Ano anterior
df_yearly['prev_win_pct'] = df_yearly.groupby('tmID')['win_pct'].shift(1).fillna(0.5)
df_yearly['trend'] = df_yearly['win_pct'] - df_yearly['prev_win_pct']

# Lifetime Win % (Histórico acumulado do treinador)
# Vamos calcular a média de win_pct até aquele ano para aquele treinador
df_yearly['lifetime_win_pct'] = df_yearly.groupby('coachID')['win_pct'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.5)

# Postseason Success (Teve sucesso recente?)
# Média móvel de vitórias nos playoffs nos últimos 2 anos
df_yearly['recent_playoff_success'] = df_yearly.groupby('coachID')['post_wins'].transform(lambda x: x.rolling(2, min_periods=1).mean()).fillna(0)

# =========================================================
# 3. TREINO DO MODELO
# =========================================================
features = [
    'win_pct', 
    'trend', 
    'tenure_years', 
    'lifetime_win_pct', 
    'recent_playoff_success',
    'is_new_coach' # Treinadores no 1º ano têm "lua de mel"?
]
target = 'coach_change'

# Conjunto de Treino: Anos 1 a 10
# Importante: Remover o ano 10 do treino se não tivermos dados do ano 11 para validar o target.
# Mas aqui, para o treino histórico, removemos linhas onde não sabemos quem é o próximo treinador (NaN)
train_df = df_yearly[df_yearly['year'] < PREDICTION_YEAR].dropna(subset=['next_coachID'])

# Conjunto de Previsão: Ano 11
predict_df = df_yearly[df_yearly['year'] == PREDICTION_YEAR].copy()

print(f"Training on {len(train_df)} historical seasons.")
print(f"Predicting for {len(predict_df)} active coaches in Season {PREDICTION_YEAR}.")

# Random Forest
rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42, class_weight='balanced')
rf.fit(train_df[features], train_df[target])

# =========================================================
# 4. PREVISÃO E RESULTADOS
# =========================================================

# Calcular probabilidades
probs = rf.predict_proba(predict_df[features])[:, 1]
predict_df['firing_prob'] = probs

# Definir Categorias de Risco
predict_df['Risk Status'] = pd.cut(
    predict_df['firing_prob'], 
    bins=[-0.1, 0.35, 0.65, 1.1], 
    labels=['Safe', 'In Danger', 'Hot Seat']
)

# Selecionar colunas finais
results = predict_df[['tmID', 'coachID', 'win_pct', 'tenure_years', 'trend', 'firing_prob', 'Risk Status']]
results = results.sort_values('firing_prob', ascending=False)

# Formatação para visualização
results_display = results.copy()
results_display['win_pct'] = (results_display['win_pct'] * 100).round(1).astype(str) + '%'
results_display['trend'] = (results_display['trend'] * 100).round(1).astype(str) + '%'
results_display['firing_prob'] = (results_display['firing_prob'] * 100).round(1).astype(str) + '%'

print("\n========================================================")
print("COACH HOT SEAT REPORT (Season 11 Forecast)")
print("========================================================")
print(results_display.to_string(index=False))

# Guardar CSV
csv_out = f"{output_dir}/season_{PREDICTION_YEAR}_coach_changes.csv"
results.to_csv(csv_out, index=False)
print(f"\n--> Detailed predictions saved to: {csv_out}")

# =========================================================
# 5. VISUALIZAÇÃO
# =========================================================
plt.figure(figsize=(10, 6))
colors = {'Safe': 'green', 'In Danger': 'orange', 'Hot Seat': 'red'}

sns.barplot(
    x='firing_prob', 
    y='coachID', 
    data=results, 
    hue='Risk Status', 
    palette=colors, 
    dodge=False
)

plt.axvline(0.5, color='black', linestyle='--', alpha=0.3)
plt.title(f'Likelihood of Coaching Change After Season {PREDICTION_YEAR}\n(Based on Predicted Team Performance)')
plt.xlabel('Probability of Change (0-1)')
plt.ylabel('Head Coach')
plt.xlim(0, 1)
plt.legend(title='Risk Level', loc='lower right')
plt.tight_layout()

plot_path = f"{plot_dir}/coach_hot_seat_s{PREDICTION_YEAR}.png"
plt.savefig(plot_path)
print(f"--> Chart saved to: {plot_path}")

# Feature Importance
print("\nModel Insights - What gets coaches fired?")
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(imp.to_string())
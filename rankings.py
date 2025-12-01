# ==============================================
# Phase 3: Regular Season Team Rankings by Conference
# ==============================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# =========================================================
# STEP 3.1: LOAD DATA
# =========================================================
data_path = "data/cleaned/full_dataset_prepared.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# =========================================================
# STEP 3.2: TEAM-LEVEL AGGREGATION
# =========================================================
if 'lgID_x' in df.columns:
    df.rename(columns={'lgID_x': 'conference'}, inplace=True)
elif 'lgID_y' in df.columns:
    df.rename(columns={'lgID_y': 'conference'}, inplace=True)
elif 'lgID' in df.columns:
    df.rename(columns={'lgID': 'conference'}, inplace=True)

team_df = df.groupby(['year', 'tmID', 'conference'], as_index=False).agg({
    'points': 'sum',
    'rebounds': 'sum',
    'assists': 'sum',
    'steals': 'sum',
    'blocks': 'sum',
    'turnovers': 'sum',
    'PF': 'sum',
    'fgMade': 'sum',
    'fgAttempted': 'sum',
    'win_pct':'max'
})

# =========================================================
# STEP 3.3: FEATURE ENGINEERING
# =========================================================
team_df['shooting_efficiency'] = team_df['fgMade'] / team_df['fgAttempted'].replace(0,1)
team_df['turnover_ratio'] = team_df['assists'] / team_df['turnovers'].replace(0,1)
team_df['def_impact'] = team_df['steals'] + team_df['blocks']
team_df['foul_discipline'] = team_df['PF']

feature_cols = [
    'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
    'shooting_efficiency', 'turnover_ratio', 'def_impact', 'win_pct'
]

# =========================================================
# STEP 3.4: LAG GENERATION
# =========================================================
print("\n Generating Lag Features (Year N-1 -> Year N)...")
team_df = team_df.sort_values(['tmID', 'year'])

for col in feature_cols:
    team_df[f'prev_{col}'] = team_df.groupby('tmID')[col].shift(1)

model_df = team_df.dropna(subset=[f'prev_{col}' for col in feature_cols])
print(f"Dataset reduced from {len(team_df)} to {len(model_df)} rows after lag creation")

# =========================================================
# STEP 3.5: MODEL TRAINING
# =========================================================
X = model_df[[f'prev_{col}' for col in feature_cols]]
y = model_df['win_pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance: MAE={mae:.4f}, R2={r2:.4f}")

# =========================================================
# STEP 3.6: MODEL PERFORMANCE PLOT
# =========================================================
plot_dir = "results/plots"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(8,8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
plt.plot([0,1], [0,1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Win %')
plt.ylabel('Predicted Win %')
plt.title(f'Model Accuracy: Actual vs Predicted (MAE: {mae:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_model_performance.png")
plt.close()
print(f"Saved model performance plot to {plot_dir}/ranking_model_performance.png")

# =========================================================
# STEP 3.7: FEATURE IMPORTANCE
# =========================================================
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', legend=False, palette='viridis')
plt.title('Feature Importance: Drivers of Next Season Success')
plt.tight_layout()
plt.savefig(f"{plot_dir}/ranking_feature_importance.png")
plt.close()
print(f"Saved feature importance plot to {plot_dir}/ranking_feature_importance.png")

# =========================================================
# STEP 3.8: PREDIÇÃO DO ANO 10
# =========================================================
target_year = 10
previous_year = target_year - 1

stats_prev_year = team_df[team_df['year'] == previous_year].copy()
actual_year_data = team_df[team_df['year'] == target_year][['tmID','conference','win_pct']].copy()
actual_year_data.rename(columns={'win_pct':'actual_win_pct'}, inplace=True)

pred_features = {f'prev_{col}': stats_prev_year[col].values for col in feature_cols}
X_future = pd.DataFrame(pred_features)

stats_prev_year['predicted_win_pct'] = model.predict(X_future)
stats_prev_year['predicted_wins'] = (stats_prev_year['predicted_win_pct'] * 34).round(0)

# =========================================================
# STEP 3.9: CREATE RANKINGS (Predicted + Actual with desempate)
# =========================================================
# Ordena por predicted_wins DESC e desempata pelo predicted_win_pct DESC
stats_prev_year = stats_prev_year.sort_values(
    ['conference', 'predicted_wins', 'predicted_win_pct'],
    ascending=[True, False, False]
)

# Calcula predicted_rank (Opção B) com desempate pelo predicted_win_pct usando 'first'
stats_prev_year['predicted_rank'] = stats_prev_year.groupby('conference')['predicted_wins'] \
    .rank(method='first', ascending=False).astype(int)

# Merge com actual
comparison_df = stats_prev_year[['tmID','conference','predicted_win_pct','predicted_wins','predicted_rank']].merge(
    actual_year_data,
    on=['tmID','conference'],
    how='left'
)

comparison_df['actual_wins'] = (comparison_df['actual_win_pct'] * 34).round(0)

# Ranking real (Opção A)
comparison_df['actual_rank'] = comparison_df.groupby('conference')['actual_wins'] \
    .rank(method='min', ascending=False).fillna(0).astype(int)

# Diferença de ranking
comparison_df['rank_diff'] = comparison_df['predicted_rank'] - comparison_df['actual_rank']

# Ordena por conferência e predicted_rank
comparison_df = comparison_df.sort_values(['conference','predicted_rank'])

print("\n📊 Comparação Ano 10 (Predicted vs Actual):")
print(comparison_df[['tmID','conference','predicted_wins','actual_wins','predicted_rank','actual_rank','rank_diff']])

# =========================================================
# STEP 3.10: PLOT PREDICTED VS ACTUAL
# =========================================================
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=comparison_df['actual_win_pct'],
    y=comparison_df['predicted_win_pct'],
    hue=comparison_df['conference'],
    s=120
)
plt.plot([0,1], [0,1], 'r--', label='Perfect Fit Line (y = x)')
plt.xlabel('Actual Win % (Season 10)')
plt.ylabel('Predicted Win % (Season 10)')
plt.title('Season 10 – Predicted vs Actual Win %')
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_pred_vs_actual.png")
plt.close()
print(f"\n📈 Gráfico guardado em: {plot_dir}/season10_pred_vs_actual.png")

# =========================================================
# STEP 3.11: SAVE RESULTS
# =========================================================
# Ranking único (Opção B)
ranking_cols = ['conference','tmID','predicted_wins','predicted_win_pct','predicted_rank']
comparison_df[ranking_cols].to_csv("results/predicted_season10_rankings.csv", index=False)

# Full comparison (Opção A)
full_cols = ['conference','tmID','predicted_win_pct','predicted_wins','actual_win_pct','actual_wins','predicted_rank','actual_rank','rank_diff']
comparison_df[full_cols].to_csv("results/season10_comparison_full.csv", index=False)

print("\n💾 Saved Season 10 Rankings and full comparison CSVs")



# =========================================================
# STEP 3.12: PLOT RANK_DIFF POR CONFERÊNCIA
# =========================================================
plt.figure(figsize=(12,6))

# Cores: verde se previsão foi melhor que real (rank_diff < 0), vermelho se pior (rank_diff > 0)
comparison_df['color'] = comparison_df['rank_diff'].apply(lambda x: 'green' if x < 0 else ('red' if x > 0 else 'gray'))

sns.barplot(
    x='tmID',
    y='rank_diff',
    hue='conference',
    data=comparison_df,
    dodge=False,
    palette=comparison_df['color'].to_list()
)

plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Team')
plt.ylabel('Predicted Rank - Actual Rank')
plt.title('Season 10 – Rank Difference (Predicted vs Actual)')
plt.legend(title='Conference')
plt.tight_layout()
plt.savefig(f"{plot_dir}/season10_rank_diff.png")
plt.close()

print(f"\n📈 Gráfico de rank_diff guardado em: {plot_dir}/season10_rank_diff.png")


# =========================================================
# DONE
# =========================================================
print("\n✅ Completed: Season 10 prediction + real comparison + rankings improved (desempate pelo predicted_win_pct).")

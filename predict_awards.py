import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12,6)

print("="*80)
print("PHASE 5: INDIVIDUAL AWARDS PREDICTION (TEST SEASON)")
print("="*80)

# =========================================================
# 1. LOAD DATA
# =========================================================
clean_path = "data/cleaned/full_dataset_prepared.csv"
awards_path = "data/awards_players.csv"

if not os.path.exists(clean_path):
    raise FileNotFoundError(f"File not found: {clean_path}")

df = pd.read_csv(clean_path)
awards_df = pd.read_csv(awards_path)

df['playerID'] = df['playerID'].astype(str).str.upper()
awards_df['playerID'] = awards_df['playerID'].astype(str).str.upper()

print(f"Data Loaded. PLayers/Teams: {df.shape}, Awards: {awards_df.shape}")

# =========================================================
# 2. DATA AGGREGATION & CLEANING
# =========================================================
metric_cols = ['GP', 'minutes', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'PF', 'fgMade', 'fgAttempted']

player_stats = df.groupby(['playerID', 'year'], as_index=False).agg({
    **{col: 'sum' for col in metric_cols},
    'win_pct': 'mean',
    'experience_years': 'max',
    'GS': 'sum'
})

player_stats['efficiency'] = (
    player_stats['points'] + player_stats['rebounds'] + player_stats['assists'] + player_stats['steals'] + player_stats['blocks'] - player_stats['turnovers'] - (player_stats['fgAttempted'] - player_stats['fgMade'])
)

for col in ['points', 'rebounds', 'assists', 'steals', 'blocks']:
    player_stats[f'{col}_pg'] = player_stats[col] / player_stats['GP'].replace(0,1)

player_stats.sort_values(['playerID', 'year'], inplace=True)
player_stats['prev_efficiency'] = player_stats.groupby('playerID')['efficiency'].shift(1).fillna(0)
player_stats['efficiency_delta'] = player_stats['efficiency'] - player_stats['prev_efficiency']

target_awards = [
    'Most Valuable Player',
    'Rookie of the Year',
    'Defensive Player of the Year',
    'Sixth Woman of the Year',
    'Most Improved Player'
]

print("\n Mapping targets for awards...")
for award in target_awards:
    winners = awards_df[awards_df['award'] == award][['playerID', 'year']]
    winners['is_winner'] = 1

    col_name = f"target_{award.replace(' ', '_')}"
    player_stats = player_stats.merge(winners, on=['playerID', 'year'], how='left')
    player_stats.rename(columns={'is_winner': col_name}, inplace=True)
    player_stats[col_name] = player_stats[col_name].fillna(0)

total_winners = player_stats[[c for c in player_stats.columns if 'target_' in c]].sum().sum()
if total_winners == 0:
    print("WARNING: No award winners matched! Check ID formatting.")
else:
    print(f"Successfully mapped {int(total_winners)} historical awards")

# =========================================================
# 3. DEFINE TRAINING LOGIC
# =========================================================
TEST_YEAR = 10
train_data = player_stats[player_stats['year'] < TEST_YEAR].fillna(0)
test_data = player_stats[player_stats['year'] == TEST_YEAR].fillna(0)

print(f"\n Training Data: {train_data['year'].nunique()} seasons ({len(train_data)} rows)")
print(f"Test Data: Season {TEST_YEAR} ({len(test_data)} rows)")

def train_and_predict(award_name, feature_cols, filter_condition=None):
    target_col = f"target_{award_name.replace(' ', '_')}"

    if filter_condition:
        train_cand = train_data[filter_condition(train_data)]
        test_cand = test_data[filter_condition(test_data)]
    else:
        train_cand = train_data
        test_cand = test_data

    X_train = train_cand[feature_cols]
    y_train = train_cand[target_col]
    X_test = test_cand[feature_cols]

    if len(y_train.unique()) < 2:
        print(f"Skipping {award_name}: Not enough history (No winners found in training set).")
        return None, []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')    
    clf.fit(X_train_scaled, y_train)

    probs = clf.predict_proba(X_test_scaled)[:, 1]

    test_cand = test_cand.copy()
    test_cand['probability'] = probs
    test_cand['predicted_rank'] = test_cand['probability'].rank(ascending=False)

    winner = test_cand.sort_values('probability', ascending=False).head(1)

    actual_winner_row = test_cand[test_cand[target_col] == 1]
    actual_winner = actual_winner_row.iloc[0]['playerID'] if not actual_winner_row.empty else "None"
    pred_winner = winner.iloc[0]['playerID'] if not winner.empty else "None"

    print(f"\n Award: {award_name}")
    print(f"Predicted Winner: {pred_winner} (Prob: {winner.iloc[0]['probability']:.3f})")
    print(f"Actual Winner: {actual_winner}")

    return clf, X_train.columns

# =========================================================
# 4. EXECUTE PREDICTIONS
# =========================================================
mvp_features = ['points_pg', 'efficiency', 'win_pct', 'minutes', 'rebounds_pg', 'assists_pg']
mvp_model, mvp_cols = train_and_predict('Most Valuable Player', mvp_features)

roy_features = ['points_pg', 'rebounds_pg', 'efficiency', 'minutes', 'GP']
train_and_predict('Rookie of the Year', roy_features, lambda x: x['experience_years'] == 0)

dpoy_features = ['steals_pg', 'blocks_pg', 'rebounds_pg', 'def_impact_proxy']

train_data['def_impact_proxy'] = train_data['steals']*2 + train_data['blocks']*2 + train_data['rebounds']
test_data['def_impact_proxy'] = test_data['steals']*2 + test_data['blocks']*2 + test_data['rebounds']
train_and_predict('Defensive Player of the Year', dpoy_features)

six_features = ['points_pg', 'efficiency', 'minutes']

filter_6th = lambda x: (x['GS'] / x['GP'].replace(0,1)) < 0.5
train_and_predict('Sixth Woman of the Year', six_features, filter_6th)

mip_features = ['efficiency_delta', 'points_pg', 'efficiency', 'minutes']

train_and_predict('Most Improved Player', mip_features, lambda x: x['experience_years'] > 0)

# =========================================================
# 5. VISUALIZATIONS
# =========================================================
plot_dir = "results/plots"
os.makedirs(plot_dir, exist_ok=True)

importances = pd.DataFrame({
    'Feature': mvp_cols,
    'Importance': mvp_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='magma', hue='Feature', legend=False)
plt.title('Feature Importance: What drives an MVP win?')
plt.tight_layout()
plt.savefig(f"{plot_dir}/mvp_feature_importance.png")
print(f"\nSaved MVP feature importance plot to {plot_dir}/mvp_feature_importance.png")

print("\n Award prediction complete.")
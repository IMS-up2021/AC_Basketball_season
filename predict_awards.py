import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14,10)

print("="*80)
print("PHASE 5: INDIVIDUAL AWARDS PREDICTION (TEST SEASON)")
print("="*80)

# =========================================================
# 1. LOAD DATA
# =========================================================
clean_path = "data/cleaned/full_dataset_prepared.csv"
awards_path = "data/awards_players.csv"
output_dir = "results"
plot_dir = "results/plots"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

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
player_stats['starter_ratio'] = player_stats['GS'] / player_stats['GP'].replace(0,1)

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

prediction_summary = []
race_data = {}

def predict_award(award_name, features, filter_func=None):
    target_col = f"target_{award_name.replace(' ', '_')}"

    if filter_func:
        train_set = train_data[filter_func(train_data)]
        test_set = test_data[filter_func(test_data)]
    else:
        train_set = train_data
        test_set = test_data
    
    if train_set[target_col].sum() == 0:
        return None

    X_train = train_set[features]
    y_train = train_set[target_col]
    X_test = test_set[features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')    
    clf.fit(X_train_scaled, y_train)

    if len(clf.classes_) > 1:
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = np.zeros(len(X_test))

    test_set = test_set.copy()
    test_set['probability'] = probs

    top_candidates = test_set.sort_values('probability', ascending=False).head(5)

    winner_row = top_candidates.iloc[0]
    actual_row = test_set[test_set[target_col] == 1]
    actual_winner = actual_row.iloc[0]['playerID'] if not actual_row.empty else "None"

    prediction_summary.append({
        'Award': award_name,
        'Predicted Winner': winner_row['playerID'],
        'Confidence': f"{winner_row['probability']:.1%}",
        'Actual Winner': actual_winner,
        'Correct': "YES" if winner_row['playerID'] == actual_winner else "NO"
    })

    race_data[award_name] = top_candidates[['playerID', 'probability']]

    print(f"Computed: {award_name}")

    return clf

# =========================================================
# 4. EXECUTE PREDICTIONS
# =========================================================
mvp_cols = ['points_pg', 'efficiency', 'win_pct', 'minutes', 'rebounds_pg', 'assists_pg']
mvp_model = predict_award('Most Valuable Player', mvp_cols)

roy_cols = ['points_pg', 'rebounds_pg', 'efficiency', 'minutes', 'GP']
predict_award('Rookie of the Year', roy_cols, lambda x: x['experience_years'] == 0)

train_data['def_score'] = train_data['steals']*2 + train_data['blocks']*2 + train_data['rebounds']
test_data['def_score'] = test_data['steals']*2 + test_data['blocks']*2 + test_data['rebounds']
dpoy_cols = ['steals_pg', 'blocks_pg', 'rebounds_pg', 'def_score']
predict_award('Defensive Player of the Year', dpoy_cols)

six_cols = ['points_pg', 'efficiency', 'minutes', 'starter_ratio']
predict_award('Sixth Woman of the Year', six_cols, lambda x: x['starter_ratio'] < 0.5)

mip_features = ['efficiency_delta', 'points_pg', 'efficiency', 'minutes']
predict_award('Most Improved Player', mip_features, lambda x: x['experience_years'] > 0)

results_df = pd.DataFrame(prediction_summary)
csv_path = f"{output_dir}/season_11_award_predictions.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
print(results_df)

# =========================================================
# 5. VISUALIZATIONS
# =========================================================
num_awards = len(race_data)
fig, axes = plt.subplots(2,3, figsize=(18,10))
axes = axes.flatten()

for i, (award, data) in enumerate(race_data.items()):
    ax = axes[i]
    sns.barplot(x='probability', y='playerID', data=data, ax=ax, palette='viridis', hue='playerID', legend=False)
    ax.set_title(award, fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("")
    ax.set_xlim(0,1)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
viz_path = f"{plot_dir}/award_races_Season_11.png"
plt.savefig(viz_path)
print(f"Visualization saved to: {viz_path}")

if mvp_model:
    plt.figure(figsize=(10,6))
    importances = pd.DataFrame({
        'Feature': mvp_cols,
        'Importance': mvp_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=importances, palette='magma', hue='Feature', legend=False)
    plt.title('What drives MVP Predictions?', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mvp_factors.png")
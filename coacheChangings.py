import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#Setup
sns.set_style("whitegrid")
os.makedirs("results/predictions", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# =========================================================
# STEP 1: LOAD DATA
# =========================================================
print("Loading data...")
try:
    players_df = pd.read_csv("data/players_teams.csv")
    coaches_df = pd.read_csv("data/cleaned/coaches_cleaned.csv")
    print(f"Coaches data: {coaches_df.shape}")
    print(f"Players data: {players_df.shape}")
except FileNotFoundError as e:
    print(" ERROR: File not found. {e}")
    exit()

# =========================================================
# STEP 2: FEATURE ENGINEERING
# =========================================================
print("Engineering Features...")

coaches_df = coaches_df.sort_values(['tmID', 'year', 'stint'])
model_df = coaches_df.drop_duplicates(subset=['tmID', 'year'], keep='last').copy()

model_df.sort_values(['tmID', 'year'], inplace=True)
model_df['next_coachID'] = model_df.groupby('tmID')['coachID'].shift(-1)

model_df['coach_change'] = (model_df['coachID'] != model_df['next_coachID']).astype(float)

model_df['league_rank'] = model_df.groupby('year')['win_pct'].rank(ascending=False, method='min')

model_df['prev_win_pct'] = model_df.groupby('tmID')['win_pct'].shift(1)
model_df['win_pct_delta'] = model_df['win_pct'] - model_df['prev_win_pct']
model_df['win_pct_delta'] = model_df['win_pct_delta'].fillna(0)

print("Calculating Roster Churn...")
p_minutes = players_df.groupby(['year', 'tmID', 'playerID'])['minutes'].sum().reset_index()

churn_map = {}
teams = p_minutes['tmID'].unique()
years = sorted(p_minutes['year'].unique())

for team in teams:
    team_data = p_minutes[p_minutes['tmID'] == team]
    for year in years:
        if year == 1:
            churn_map[(team, year)] = 0
            continue
        
        curr_roster = team_data[team_data['year'] == year]
        prev_roster = team_data[team_data['year'] == year - 1]

        if prev_roster.empty or curr_roster.empty:
            churn_map[(team, year)] = 0.5
            continue

        prev_players = set(prev_roster['playerID'])
        curr_total_minutes = curr_roster['minutes'].sum()

        if curr_total_minutes == 0:
            churn_val = 0
        else:
            returning_minutes = curr_roster[curr_roster['playerID'].isin(prev_players)]['minutes'].sum()
            stability = returning_minutes / curr_total_minutes
            churn_val = 1 - stability

        churn_map[(team, year)] = churn_val

model_df['roster_churn'] = model_df.apply(lambda x: churn_map.get((x['tmID'], x['year']), 0), axis=1)

# =========================================================
# STEP 3: PREPARE MODEL
# =========================================================

features = [
    'win_pct',
    'league_rank',
    'win_pct_delta',
    'tenure_years',
    'roster_churn',
    'won',
    'lost'
]
target = 'coach_change'

latest_season = model_df['year'].max()

train_df = model_df[model_df['year'] < latest_season].copy()
train_df = train_df.dropna(subset=[target])

test_df = model_df[model_df['year'] == latest_season].copy()

train_df[features] = train_df[features].fillna(0)
test_df[features] = test_df[features].fillna(0)

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target] if not test_df[target].isna().all() else None

print(f"Training Data: {X_train.shape[0]} samples")
print(f"Test Data: {X_test.shape[0]} samples (Year {int(latest_season)})")

# =========================================================
# STEP 4: TRAINING 
# =========================================================
print("\n Training Random Forest Classifier...")

rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', max_depth=6, min_samples_leaf=4)
rf.fit(X_train, y_train)

print("Model trained successfully!")

print("\n Evaluating model performance...")

y_prob = rf.predict_proba(X_test)[:, 1]

THRESHOLD = 0.45
y_pred_custom = (y_prob >= THRESHOLD).astype(int)

# =========================================================
# STEP 5: EVALUATION
# =========================================================
print("\n--- Confusion Matrix ---")
plt.figure(figsize=(6,5))

cm = confusion_matrix(y_test, y_pred_custom, labels=[0,1])

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Safe', 'Fired'], yticklabels=['Safe', 'Fired'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix (Threshold {THRESHOLD})')
plt.tight_layout()
plt.savefig("results/plots/coach_churn_confusion_matrix.png")
plt.show()

print("--- Classification Report ---")
print(classification_report(y_test, y_pred_custom,labels=[0,1], target_names=['Safe', 'Fired'], zero_division=0))

if len(np.unique(y_test)) > 1:
    auc = roc_auc_score(y_test, y_prob)
    print(f"--- AUC-ROC Score ---")
    print(f"AUC-ROC: {auc:.4f}\n")
else:
    print(f"--- AUC-ROC Score: N/A (Test set contains only one class) ---")

importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
plt.title('Feature Importance: Coach Changes')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("results/plots/coach_firing_factors.png")
plt.show()

# =========================================================
# STEP 6: SAVE PREDICTIONS
# =========================================================
results = test_df.copy()
results['firing_probability'] = y_prob
results['predicted_change'] = y_pred_custom

hot_seat = results[results['predicted_change'] == 1][['tmID', 'coachID', 'win_pct', 'league_rank', 'win_pct_delta', 'firing_probability']]
hot_seat = hot_seat.sort_values('firing_probability', ascending=False)

print("\nCOACHES ON THE HOT SEAT (Predicted Change):")
print(hot_seat)

hot_seat.to_csv("results/predictions/season_10_coach_changes.csv", index=False)
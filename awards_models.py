# ==============================================
# Phase 5: Task 3 – Individual Awards (Steps 5.2–5.3)
# Model Development and Prediction for Season 11
# ==============================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("PHASE 5: AWARD MODEL DEVELOPMENT AND SEASON 11 PREDICTION")
print("=" * 80)

# =========================================================
# STEP 5.1: LOAD DATA
# =========================================================
awards_path = os.path.join("data/", "awards_players.csv")
players_teams_path = os.path.join("data/", "players_teams.csv")
players_path = os.path.join("data/cleaned/", "players_cleaned.csv")

for p in [awards_path, players_teams_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ Missing required file: {p}")

awards = pd.read_csv(awards_path)
players_teams = pd.read_csv(players_teams_path)
players = pd.read_csv(players_path) if os.path.exists(players_path) else None

print(f"✅ Loaded datasets: awards={awards.shape}, player_team={players_teams.shape}")
print("Awards columns:", awards.columns.tolist())

# Detect correct column name for award
award_col_candidates = [c for c in awards.columns if 'award' in c.lower()]
if not award_col_candidates:
    raise KeyError("❌ No column containing 'award' found in awards_players.csv")
award_col = award_col_candidates[0]

# =========================================================
# STEP 5.2: FEATURE ENGINEERING
# =========================================================
print("\n🧩 Preparing features...")

numeric_cols = players_teams.select_dtypes(include=[np.number]).columns.tolist()
stats = players_teams.groupby(['playerID', 'year'], as_index=False)[numeric_cols].mean()

# Merge award info
award_flags = awards.groupby(['playerID', 'year'])[award_col].unique().reset_index()
stats = stats.merge(award_flags, on=['playerID', 'year'], how='left')
stats[award_col] = stats[award_col].apply(lambda x: x if isinstance(x, list) else [])

award_types = sorted(awards[award_col].dropna().unique())
print(f"🏆 Found {len(award_types)} distinct awards.")

# =========================================================
# STEP 5.3: MODEL TRAINING & PREDICTION
# =========================================================
os.makedirs("results/predictions", exist_ok=True)
os.makedirs("data/cleaned", exist_ok=True)

results_summary = []

for award in award_types:
    print(f"\n=== Training model for {award} ===")

    stats[f"is_{award}"] = stats[award_col].apply(lambda x: 1 if award in x else 0)
    df = stats.drop(columns=[award_col]).fillna(0)

    train_df = df[df['year'] < 11]
    test_df = df[df['year'] == 11]

    if train_df[f"is_{award}"].sum() < 3 or test_df.empty:
        print(f"⚠️ Skipping {award} — insufficient data for modeling or missing year 11.")
        continue

    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[f"is_{award}"], errors='ignore')
    y_train = train_df[f"is_{award}"]

    X_test = test_df.select_dtypes(include=[np.number]).drop(columns=[f"is_{award}"], errors='ignore')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight='balanced_subsample'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on training data (since we skip validation here)
    y_pred_train = model.predict(X_train_scaled)
    acc = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    results_summary.append({
        'award': award,
        'accuracy_train': acc,
        'f1_train': f1,
        'n_train_samples': len(train_df),
        'n_winners_train': y_train.sum()
    })
    print(f"✅ {award}: acc={acc:.3f}, f1={f1:.3f}")

    # Predict year 11
    test_df['prediction_prob'] = model.predict_proba(X_test_scaled)[:, 1]
    test_df['predicted_winner'] = (test_df['prediction_prob'] > 0.5).astype(int)

    predictions = test_df[['playerID', 'year', 'prediction_prob', 'predicted_winner']]
    predictions = predictions.sort_values(by='prediction_prob', ascending=False)

    output_pred_path = f"results/predictions/{award}_season_11_predictions.csv"
    predictions.to_csv(output_pred_path, index=False)
    print(f"📄 Saved season 11 predictions for {award} → {output_pred_path}")

# =========================================================
# STEP 5.4: SAVE SUMMARY
# =========================================================
summary_path = "data/cleaned/award_models_summary.csv"
pd.DataFrame(results_summary).to_csv(summary_path, index=False)
print(f"\n✅ Saved model performance summary to {summary_path}")

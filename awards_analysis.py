# ==============================================
# Phase 5: Task 3 – Individual Awards (Step 5.1)
# Award Categories Analysis
# ==============================================

import pandas as pd
import os

print("=" * 80)
print("PHASE 5: INDIVIDUAL AWARDS ANALYSIS")
print("=" * 80)

# =========================================================
# STEP 5.1: LOAD DATA
# =========================================================
awards_path = os.path.join("data/", "awards_players.csv")
players_path = os.path.join("data/cleaned/", "players_cleaned.csv")
teams_path = os.path.join("data/", "teams.csv")

if not os.path.exists(awards_path):
    raise FileNotFoundError(f"❌ Missing file: {awards_path}")

awards = pd.read_csv(awards_path)
players = pd.read_csv(players_path) if os.path.exists(players_path) else None
teams = pd.read_csv(teams_path) if os.path.exists(teams_path) else None

print(f"✅ Loaded awards data: {awards.shape[0]} records")
print("Columns:", awards.columns.tolist())

# =========================================================
# STEP 5.2: IDENTIFY AWARD TYPES
# =========================================================
print("\n🏆 Identifying award categories...")

# Detect column name automatically
award_col_candidates = [c for c in awards.columns if 'award' in c.lower()]
if award_col_candidates:
    award_col = award_col_candidates[0]
else:
    raise KeyError("❌ Could not find a column containing award names in awards_players.csv")

award_counts = awards[award_col].value_counts().reset_index()
award_counts.columns = ['award', 'count']
print(award_counts)

# =========================================================
# STEP 5.3: ANALYZE HISTORICAL PROFILES
# =========================================================
print("\n📊 Analyzing historical award winners...")

# Merge player data if available
if players is not None and 'playerID' in players.columns:
    merged = awards.merge(players, on='playerID', how='left')
else:
    merged = awards.copy()

# Basic descriptive stats
award_summary = merged.groupby(award_col).agg({
    'year': ['min', 'max', 'nunique'],
    'playerID': 'nunique'
}).reset_index()

award_summary.columns = ['award', 'first_year', 'last_year', 'n_years', 'unique_winners']
print("\n📈 Summary by award:")
print(award_summary)

# Save summary
output_path = "data/cleaned/award_categories_summary.csv"
award_summary.to_csv(output_path, index=False)
print(f"\n✅ Saved award category summary to {output_path}")

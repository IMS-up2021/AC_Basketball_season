import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# VISUALIZATION STYLE
# ----------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PHASE 1: DATA UNDERSTANDING & EXPLORATION")
print("Basketball Season Prediction Project")
print("="*80)

# ============================================================================
# STEP 1.1: ENVIRONMENT SETUP & DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("STEP 1.1: LOADING DATASETS")
print("="*80)

# Define base path
DATA_PATH = "data/"

# Load all datasets from 'data' folder
awards_players = pd.read_csv(f"{DATA_PATH}awards_players.csv")
coaches = pd.read_csv(f"{DATA_PATH}coaches.csv")
players = pd.read_csv(f"{DATA_PATH}players.csv")
players_teams = pd.read_csv(f"{DATA_PATH}players_teams.csv")
series_post = pd.read_csv(f"{DATA_PATH}series_post.csv")
teams = pd.read_csv(f"{DATA_PATH}teams.csv")
teams_post = pd.read_csv(f"{DATA_PATH}teams_post.csv")

datasets = {
    'awards_players': awards_players,
    'coaches': coaches,
    'players': players,
    'players_teams': players_teams,
    'series_post': series_post,
    'teams': teams,
    'teams_post': teams_post
}

print("\n✓ All datasets loaded successfully!\n")
print("Dataset Dimensions:")
for name, df in datasets.items():
    print(f"  {name:20s}: {df.shape[0]:6d} rows × {df.shape[1]:3d} columns")

# ============================================================================
# STEP 1.2: INITIAL DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("STEP 1.2: INITIAL DATA EXPLORATION")
print("="*80)

# --- AWARDS_PLAYERS Dataset ---
print("\n" + "-"*80)
print("1. AWARDS_PLAYERS Dataset")
print("-"*80)
print("\nFirst 5 rows:")
print(awards_players.head())
print("\nData Types:")
print(awards_players.dtypes)
print("\nMissing Values:")
print(awards_players.isnull().sum())
print("\nUnique Values:")
print(f"  Unique Players: {awards_players['playerID'].nunique()}")
print(f"  Unique Awards: {awards_players['award'].nunique()}")
print(f"  Years Range: {awards_players['year'].min()} to {awards_players['year'].max()}")
print(f"  Leagues: {awards_players['lgID'].unique()}")

print("\nAward Types:")
award_counts = awards_players['award'].value_counts()
for award, count in award_counts.items():
    print(f"  {award:45s}: {count:3d} occurrences")

# --- COACHES Dataset ---
print("\n" + "-"*80)
print("2. COACHES Dataset")
print("-"*80)
print("\nFirst 5 rows:")
print(coaches.head())
print("\nData Types:")
print(coaches.dtypes)
print("\nMissing Values:")
print(coaches.isnull().sum())
print("\nBasic Statistics:")
print(coaches[['won', 'lost', 'post_wins', 'post_losses']].describe())
print(f"\nUnique Coaches: {coaches['coachID'].nunique()}")
print(f"Unique Teams: {coaches['tmID'].nunique()}")
print(f"Years Range: {coaches['year'].min()} to {coaches['year'].max()}")
print(f"Teams: {sorted(coaches['tmID'].unique())}")

# --- PLAYERS Dataset ---
print("\n" + "-"*80)
print("3. PLAYERS Dataset")
print("-"*80)
print("\nFirst 5 rows:")
print(players.head())
print("\nData Types:")
print(players.dtypes)
print("\nMissing Values:")
missing_players = players.isnull().sum()
print(missing_players[missing_players > 0])
print(f"\nTotal Players: {len(players)}")
print(f"Unique Positions: {players['pos'].unique()}")

# Position distribution
print("\nPosition Distribution:")
pos_dist = players['pos'].value_counts()
for pos, count in pos_dist.items():
    print(f"  {pos:10s}: {count:4d} players")

# Height and weight statistics (excluding missing values)
print("\nPhysical Attributes (excluding missing):")
print(f"  Height - Mean: {players['height'].mean():.1f} inches, "
      f"Range: {players['height'].min():.0f} - {players['height'].max():.0f}")
print(f"  Weight - Mean: {players['weight'].mean():.1f} lbs, "
      f"Range: {players['weight'].min():.0f} - {players['weight'].max():.0f}")

# ============================================================================
# STEP 1.3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 1.3: EXPLORATORY DATA ANALYSIS")
print("="*80)

# --- Coach Performance Analysis ---
print("\n" + "-"*80)
print("COACH PERFORMANCE ANALYSIS")
print("-"*80)

# Calculate win percentage for coaches
coaches['win_pct'] = coaches['won'] / (coaches['won'] + coaches['lost'])
coaches['games_coached'] = coaches['won'] + coaches['lost']

print("\nTop 10 Coaches by Win Percentage (min 30 games):")
top_coaches = coaches[coaches['games_coached'] >= 30].nlargest(10, 'win_pct')
for _, row in top_coaches.iterrows():
    print(f"  {row['coachID']:15s} | Year {row['year']} | "
          f"Win%: {row['win_pct']:.3f} | W-L: {row['won']}-{row['lost']}")

# Coach turnover analysis
print("\nCoach Tenure Analysis:")
coach_seasons = coaches.groupby('coachID')['year'].agg(['count', 'min', 'max'])
coach_seasons['tenure'] = coach_seasons['max'] - coach_seasons['min'] + 1
print(f"  Average seasons per coach: {coach_seasons['count'].mean():.2f}")
print(f"  Average tenure: {coach_seasons['tenure'].mean():.2f} years")
print(f"  Longest tenure: {coach_seasons['tenure'].max():.0f} years")

# --- Award Winners Analysis ---
print("\n" + "-"*80)
print("AWARD WINNERS ANALYSIS")
print("-"*80)

# Most Valuable Player winners
mvp_winners = awards_players[awards_players['award'] == 'Most Valuable Player']
print("\nMost Valuable Player Winners:")
for _, row in mvp_winners.iterrows():
    print(f"  Year {row['year']:2d}: {row['playerID']}")

# Players with multiple awards
player_awards = awards_players.groupby('playerID').size()
multi_award_winners = player_awards[player_awards > 1].sort_values(ascending=False)
print("\nTop 10 Players by Total Awards:")
for player, count in multi_award_winners.head(10).items():
    player_award_types = awards_players[awards_players['playerID'] == player]['award'].unique()
    print(f"  {player:15s}: {count:2d} awards")
    for award in player_award_types:
        award_count = len(awards_players[(awards_players['playerID'] == player) &
                                         (awards_players['award'] == award)])
        print(f"    - {award}: {award_count}x")

# --- Temporal Trends ---
print("\n" + "-"*80)
print("TEMPORAL TRENDS")
print("-"*80)

# Awards over time
print("\nAwards Distribution by Year:")
awards_by_year = awards_players.groupby('year').size()
for year, count in awards_by_year.items():
    print(f"  Year {year:2d}: {count:2d} awards")

# Coach changes per year
print("\nUnique Coaches per Year:")
coaches_by_year = coaches.groupby('year')['coachID'].nunique()
for year, count in coaches_by_year.items():
    print(f"  Year {year:2d}: {count:2d} coaches")

# ============================================================================
# SUMMARY & KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & OBSERVATIONS")
print("="*80)

print("\n1. DATA CHARACTERISTICS:")
print(f"   • League: {awards_players['lgID'].unique()[0]} (WNBA)")
print(f"   • Time Period: Years {awards_players['year'].min()}–{awards_players['year'].max()}")
print(f"   • Teams: {coaches['tmID'].nunique()} unique team codes")
print(f"   • Coaches: {coaches['coachID'].nunique()} unique coaches")
print(f"   • Players: {players['bioID'].nunique()} unique players")

print("\n2. AWARD CATEGORIES IDENTIFIED:")
awards_list = awards_players['award'].unique()
for i, award in enumerate(awards_list, 1):
    print(f"   {i}. {award}")

print("\n3. MISSING DATA ISSUES:")
print("   • Players dataset: Missing height/weight for some entries")
print("   • Coaches dataset: Some missing postseason stats (expected)")
print("   • Need to validate joins with teams and players_teams")

print("\n4. COACH TURNOVER PATTERNS:")
print(f"   • Average coach tenure: {coach_seasons['tenure'].mean():.1f} years")
print(f"   • Some coaches have multiple stints (stint column)")

print("\n5. NEXT STEPS:")
print("   • Explore relationships between datasets (player-team-season mappings)")
print("   • Build features for prediction tasks (e.g., win %, award likelihood)")
print("   • Visualize correlations and trends using matplotlib/seaborn")

print("\n" + "="*80)
print("PHASE 1 COMPLETE!")
print("="*80)

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
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

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
try:
    awards_players = pd.read_csv(f"{DATA_PATH}awards_players.csv")
    coaches = pd.read_csv(f"{DATA_PATH}coaches.csv")
    players = pd.read_csv(f"{DATA_PATH}players.csv")
    players_teams = pd.read_csv(f"{DATA_PATH}players_teams.csv")
    series_post = pd.read_csv(f"{DATA_PATH}series_post.csv")
    teams = pd.read_csv(f"{DATA_PATH}teams.csv")
    teams_post = pd.read_csv(f"{DATA_PATH}teams_post.csv")
    print("\n ✓ All datasets loaded successfully!\n")
except FileNotFoundError as e:
    print(f"\nError: {e}. Make sure the 'data' folder and CSV files exist")
    exit()

# ============================================================================
# STEP 1.2: INITIAL DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("STEP 1.2: INITIAL DATA EXPLORATION")
print("="*80)

def explore_df(df, name):
    # --- AWARDS_PLAYERS Dataset ---
    print("\n" + "-"*80)
    print(f"Exploring: {name} Dataset")
    print("-"*80)
    print(f"\nDimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes.to_string())
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    if missing_values.sum() == 0:
        print("No missing values found")
    else:
        print(missing_values[missing_values > 0])

explore_df(awards_players, "Awards Players")
explore_df(coaches, "Coaches")
explore_df(players, "Players")
explore_df(teams, "Teams")

# ============================================================================
# STEP 1.3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 1.3: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Merge awards with player details to analyze winners
awards_details = pd.merge(awards_players, players, left_on='playerID', right_on='bioID', how='left')

coaches_with_teams = pd.merge(coaches, teams[['tmID', 'name']], on='tmID', how='left')

print("✓ Merged 'awards' with 'players' for winner analysis")
print("✓ Merged 'coaches' with 'teams' for team name context")

print("\n" + "-"*80)
print("Exploratory data analysis & visualizations")
print("="*80)

# Award winners analysis
print("\nAnalyzing Award Winners...")

plt.figure()
award_counts = awards_details['award'].value_counts().nlargest(10)
sns.barplot(x=award_counts.values, y=award_counts.index, palette='viridis', orient='h')
plt.title('Top 10 Most Frequent Awards')
plt.xlabel('Number of Times Awarded')
plt.ylabel('Award')
plt.tight_layout()
plt.show()

# Analyze the positions of MVP winners
mvp_winners = awards_details[awards_details['award'] == 'Most Valuable Player']
mvp_pos_counts = mvp_winners['pos'].value_counts()

plt.figure()
sns.barplot(x=mvp_pos_counts.index, y=mvp_pos_counts.values, palette='plasma')
plt.title('Position Distribution of MVP Winners')
plt.xlabel('Player Position')
plt.ylabel('Number of MVP Awards')
plt.show()

print("\nMost Decorated Players (by total awards):")
player_award_counts = awards_details['playerID'].value_counts().nlargest(5)
print(player_award_counts)

# Coach performance analysis
print("\nAnalyzing Coach Performance...")
coaches['win_pct'] = coaches['won'] / (coaches['won'] + coaches['lost'])
coaches['games_coached'] = coaches['won'] + coaches['lost']

top_coaches = coaches[coaches['games_coached'] >= 34].nlargest(10, 'win_pct')

plt.figure()
sns.barplot(x='win_pct', y='coachID', data=top_coaches, palette='magma')
plt.title('Top 10 Coaches by Win Percentage (min 34 games)')
plt.xlabel('Win Percentage')
plt.ylabel('Coach ID')
plt.xlim(0.6, 0.8)
plt.tight_layout()
plt.show()

# Player Physical Attributes Analysis
print("\nAnalyzing Player Physical Attributes...")

fig, axes = plt.subplots(1, 2, figsize = (16,6))
sns.histplot(players['height'].dropna(), kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Player Height (inches)')
axes[0].set_xlabel('Height (inches)')

sns.histplot(players['weight'].dropna(), kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Player Weight (lbs)')
axes[1].set_xlabel('Weight (lbs)')
plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY & KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & OBSERVATIONS")
print("="*80)

print("\n1. DATA CHARACTERISTICS:")
print(f"   • The data primarly covers the {awards_players['lgID'].unique()[0]} league.")
print(f" • The analysis spans from {awards_players['year'].min()} to {awards_players['year'].max()}") 

print("\n2. PLAYER INSIGHTS:")
print(f" • The MVP award is predominantly won by players in the '{mvp_pos_counts.index[0]}' position.")
print(f" • Player heights show a normal distribution, peaking around {players['height'].median():.0f} inches")

print("\n3. COACHING PATTERNS:")
print(f"   • A small group of elite coaches consistently achieve high win percentages (often > 70%)")
coach_seasons = coaches.groupby('coachID')['year'].count()
print(f"   • The average number of seasons coached is {coach_seasons.mean():.2f}, indicating relatively high turnover")

print("\n" + "="*80)
print("PHASE 1 COMPLETE!")
print("="*80)

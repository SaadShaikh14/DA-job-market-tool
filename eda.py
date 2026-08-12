"""
eda.py
Step 4: Analytics layer. Loads the cleaned job postings and produces
the core charts: top skills overall, top skills by city, experience
level breakdown, postings-over-time trend, and a salary data summary.
Charts are saved as PNG files in a 'charts' folder.
"""

import ast
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('charts', exist_ok=True)

df = pd.read_csv('da_job_postings_clean.csv')
df['skills_matched'] = df['skills_matched'].apply(ast.literal_eval)
df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')

# Extract a clean city name (first part before the comma; "India" stays as-is)
df['city'] = df['location_clean'].apply(lambda x: str(x).split(',')[0].strip())

print("=" * 50)
print(f"Total postings: {len(df)}")
print(f"Postings with at least 1 skill matched: {(df['num_skills'] > 0).sum()}")
print("=" * 50)

# 1. Top skills overall
all_skills = [s for row in df['skills_matched'] for s in row]
skill_counts = Counter(all_skills)
top_skills = pd.Series(dict(skill_counts.most_common(15)))

print("\nTop 15 skills overall:")
print(top_skills)

plt.figure(figsize=(9, 6))
top_skills.sort_values().plot(kind='barh', color='#2E86AB')
plt.title('Top 15 Most In-Demand Skills — Data Analyst Roles (India)')
plt.xlabel('Number of postings mentioning this skill')
plt.tight_layout()
plt.savefig('charts/top_skills.png', dpi=150)
plt.close()

# 2. Top skills by top 5 cities
top_cities = df['city'].value_counts().head(5).index.tolist()
print(f"\nTop 5 cities by posting count: {top_cities}")

fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
for ax, city in zip(axes, top_cities):
    city_df = df[df['city'] == city]
    city_skills = [s for row in city_df['skills_matched'] for s in row]
    city_top = pd.Series(Counter(city_skills).most_common(6))
    if len(city_top) > 0:
        labels, values = zip(*Counter(city_skills).most_common(6))
        ax.barh(labels[::-1], values[::-1], color='#A23B72')
    ax.set_title(city)
plt.suptitle('Top Skills by City')
plt.tight_layout()
plt.savefig('charts/skills_by_city.png', dpi=150)
plt.close()

# 3. Experience level breakdown
exp_counts = df['experience_level_guess'].value_counts()
print("\nExperience level breakdown:")
print(exp_counts)

plt.figure(figsize=(7, 7))
exp_counts.plot(kind='pie', autopct='%1.0f%%', colors=['#F18F01', '#C73E1D', '#2E86AB', '#A23B72', '#6A994E'])
plt.title('Experience Level Breakdown (guessed from job title)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('charts/experience_level.png', dpi=150)
plt.close()

# 4. Postings over time
time_df = df.dropna(subset=['posted_date']).copy()
time_df['week'] = time_df['posted_date'].dt.to_period('W').dt.start_time
weekly_counts = time_df.groupby('week').size()

print("\nPostings by week:")
print(weekly_counts)

plt.figure(figsize=(10, 5))
weekly_counts.plot(kind='line', marker='o', color='#2E86AB')
plt.title('Data Analyst Job Postings Over Time')
plt.xlabel('Week')
plt.ylabel('Number of postings')
plt.tight_layout()
plt.savefig('charts/postings_over_time.png', dpi=150)
plt.close()

# 5. Salary data summary
has_salary = df['salary_min'].notna() | df['salary_max'].notna()
print(f"\nPostings with any salary data: {has_salary.sum()} / {len(df)} ({has_salary.mean()*100:.1f}%)")
if has_salary.sum() > 0:
    print(df.loc[has_salary, ['title', 'company', 'salary_min', 'salary_max']].head(10))

print("\nCharts saved in the 'charts' folder:")
print(" - top_skills.png")
print(" - skills_by_city.png")
print(" - experience_level.png")
print(" - postings_over_time.png")

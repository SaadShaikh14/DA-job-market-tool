"""
clean_and_extract.py
Cleans the raw job postings and extracts mentioned skills from each
job's title + description using keyword matching. Also adds a rough
experience-level guess based on the job title.
"""

import re
import pandas as pd

df = pd.read_csv('da_job_postings_raw.csv')

df['text_for_scan'] = (df['title'].fillna('') + ' ' + df['description'].fillna(''))

SKILLS = [
    'Python', 'SQL', 'Excel', 'Power BI', 'Tableau', 'R', 'SAS', 'SPSS',
    'VBA', 'Alteryx', 'Looker', 'Qlik', 'MySQL', 'PostgreSQL', 'MongoDB',
    'NoSQL', 'Big Data', 'Hadoop', 'Spark', 'AWS', 'Azure', 'GCP',
    'Google Cloud', 'Machine Learning', 'Statistics', 'A/B Testing',
    'Data Visualization', 'ETL', 'Data Warehousing', 'Data Modeling',
    'Git', 'JIRA', 'Snowflake', 'DAX', 'Power Query', 'Google Sheets',
    'Google Analytics',
]


def find_skills(text):
    found = []
    text_lower = text.lower()
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


df['skills_matched'] = df['text_for_scan'].apply(find_skills)
df['num_skills'] = df['skills_matched'].apply(len)
df['skills_matched_str'] = df['skills_matched'].apply(lambda x: ', '.join(x))


def guess_level(title):
    t = str(title).lower()
    if 'fresher' in t or 'trainee' in t or 'intern' in t:
        return 'Fresher/Trainee'
    if 'junior' in t or 'associate' in t:
        return 'Junior'
    if 'senior' in t or 'sr.' in t or 'sr ' in t or 'lead' in t:
        return 'Senior'
    if 'manager' in t or 'head' in t:
        return 'Manager+'
    return 'Not specified'


df['experience_level_guess'] = df['title'].apply(guess_level)
df['location_clean'] = df['location'].fillna('Not specified')

df.drop(columns=['text_for_scan'], inplace=True)
df.to_csv('da_job_postings_clean.csv', index=False)

print("Saved cleaned data to da_job_postings_clean.csv")
print()
print("Rows:", len(df))
print("Rows with at least 1 skill matched:", (df['num_skills'] > 0).sum())
print()
print("Top 15 skills overall:")
all_skills = [s for row in df['skills_matched'] for s in row]
print(pd.Series(all_skills).value_counts().head(15))
print()
print("Experience level breakdown:")
print(df['experience_level_guess'].value_counts())

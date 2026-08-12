"""
clean_data.py
Thorough data-quality pass on the raw postings before skill extraction:
- Removes duplicate postings (by URL)
- Strips messy whitespace from text fields
- Flags and removes clearly invalid salary values
- Keeps a note of which salaries are Adzuna-predicted vs employer-stated
Run this BEFORE clean_and_extract.py.
"""

import pandas as pd

df = pd.read_csv('da_job_postings_raw.csv')
print(f"Starting rows: {len(df)}")

# 1. Remove duplicate postings by URL (same job can appear across
#    multiple title searches)
before = len(df)
df.drop_duplicates(subset=['url'], inplace=True)
print(f"Removed {before - len(df)} duplicate postings (same URL)")

# 2. Clean text fields
for col in ['title', 'company', 'location']:
    df[col] = df[col].astype(str).str.strip()

# 3. Fix invalid salary values
#    - salary_max < salary_min is a data error -> drop both values
#    - values under 1000 are placeholder/garbage, not real annual salaries
invalid_range = df['salary_max'] < df['salary_min']
too_small = (df['salary_min'].notna() & (df['salary_min'] < 1000)) | \
            (df['salary_max'].notna() & (df['salary_max'] < 1000))
bad_salary = invalid_range | too_small

print(f"Clearing {bad_salary.sum()} invalid salary values (kept the posting, just blanked the salary)")
df.loc[bad_salary, ['salary_min', 'salary_max']] = None

# 4. Cap extreme outliers at the 1st/99th percentile instead of deleting
#    (keeps the postings, avoids a handful of extreme values skewing charts)
for col in ['salary_min', 'salary_max']:
    valid = df[col].dropna()
    if len(valid) > 10:
        low, high = valid.quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=low, upper=high)

df['salary_is_predicted'] = df['salary_is_predicted'].fillna(0).astype(int)

df.to_csv('da_job_postings_raw.csv', index=False)
print(f"\nFinal rows: {len(df)}")
print(f"Rows with usable salary data: {df['salary_min'].notna().sum()}")
print("Saved cleaned data back to da_job_postings_raw.csv")

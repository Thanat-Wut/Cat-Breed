import pandas as pd

df = pd.read_csv('BU Data from Survey Cases_final(5).csv', encoding='cp874', header=1)

with open('data_deep_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== Col 5-9: Purchase Decision Factors (Likert) ===\n')
    for i in range(5, 10):
        vals = df.iloc[:, i].dropna().value_counts().to_dict()
        f.write(f'Col {i}: {vals}\n')
    
    f.write('\n=== Col 12-19: Packaging Opinion Factors (Likert) ===\n')
    for i in range(12, 20):
        vals = df.iloc[:, i].dropna().value_counts().to_dict()
        f.write(f'Col {i}: {vals}\n')
    
    f.write('\n=== Col 22-26: Option 1 Ratings ===\n')
    for i in range(22, 27):
        vals = df.iloc[:, i].dropna().value_counts().to_dict()
        f.write(f'Col {i}: {vals}\n')
    
    f.write('\n=== Col 72: Top 3 Choices ===\n')
    vals = df.iloc[:, 72].dropna().value_counts().head(15).to_dict()
    f.write(f'{vals}\n')
    
    f.write('\n=== Col 73-75: Demographics ===\n')
    for i in [73, 74, 75]:
        vals = df.iloc[:, i].dropna().value_counts().to_dict()
        f.write(f'Col {i}: {vals}\n')
    
    f.write('\n=== Col 1: Experience ===\n')
    vals = df.iloc[:, 1].dropna().value_counts().to_dict()
    f.write(f'Col 1: {vals}\n')
    
    f.write('\n=== Col 10-11 ===\n')
    for i in [10, 11]:
        vals = df.iloc[:, i].dropna().value_counts().to_dict()
        f.write(f'Col {i}: {vals}\n')
    
    # Check string lengths of Likert values to identify them
    f.write('\n=== Unique Likert values (cols 5-9) by length ===\n')
    all_vals = set()
    for i in range(5, 10):
        for v in df.iloc[:, i].dropna().unique():
            all_vals.add((v, len(str(v))))
    for v, l in sorted(all_vals, key=lambda x: x[1]):
        f.write(f'  len={l}: "{v}"\n')
    
    f.write('\n=== Unique Likert values (cols 22-26 Option 1) by length ===\n')
    all_vals = set()
    for i in range(22, 27):
        for v in df.iloc[:, i].dropna().unique():
            all_vals.add((v, len(str(v))))
    for v, l in sorted(all_vals, key=lambda x: x[1]):
        f.write(f'  len={l}: "{v}"\n')

    # Check a few data rows
    f.write('\n=== Sample rows (non-null) ===\n')
    non_null_rows = df.dropna(subset=[df.columns[72]])
    for idx in non_null_rows.index[:3]:
        f.write(f'\nRow {idx}:\n')
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 72, 73, 74, 75]:
            f.write(f'  Col {i}: {df.iloc[idx, i]}\n')

print('Done')

import pandas as pd
df = pd.read_csv('BU Data from Survey Cases_final(5).csv', encoding='utf-8-sig', header=1)
with open('cat_brand_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('Breeds:\n' + '\n'.join(df.iloc[:, 3].dropna().unique().astype(str)) + '\n\n')
    f.write('Brands:\n' + '\n'.join(df.iloc[:, 4].dropna().unique().astype(str)))

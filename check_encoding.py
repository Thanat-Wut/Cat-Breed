import pandas as pd
df = pd.read_csv('BU Data from Survey Cases_final(5).csv', encoding='cp874', header=1)

print('=== Col 1 Experience ===')
print(df.iloc[:, 1].dropna().unique())

print('\n=== Col 5-9 Likert 5 ===')
for i in range(5,10):
    print(f'Col {i}:', df.iloc[:, i].dropna().unique())

print('\n=== Col 22-26 Likert 4 ===')
for i in range(22,27):
    print(f'Col {i}:', df.iloc[:, i].dropna().unique())

print('\n=== Col 73 Age ===')
print(df.iloc[:, 73].dropna().unique())

print('\n=== Col 74 Gender ===')
print(df.iloc[:, 74].dropna().unique())

print('\n=== Col 75 Marital ===')
print(df.iloc[:, 75].dropna().unique())

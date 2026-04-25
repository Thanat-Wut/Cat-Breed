import pandas as pd
df = pd.read_csv('BU Data from Survey Cases_final(5).csv', encoding='utf-8-sig', header=1)
with open('utf8_output.txt', 'w', encoding='utf-8') as f:
    f.write('Experience: ' + str(df.iloc[:, 1].dropna().unique().tolist()) + '\n')
    f.write('Likert 5: ' + str(df.iloc[:, 5].dropna().unique().tolist()) + '\n')
    f.write('Likert 4: ' + str(df.iloc[:, 22].dropna().unique().tolist()) + '\n')
    f.write('Age: ' + str(df.iloc[:, 73].dropna().unique().tolist()) + '\n')
    f.write('Gender: ' + str(df.iloc[:, 74].dropna().unique().tolist()) + '\n')
    f.write('Marital: ' + str(df.iloc[:, 75].dropna().unique().tolist()) + '\n')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import plots as pl
import math

df = pd.read_excel('data.xlsx')

data = df.iloc[:8, :5]
data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
teamNames = pd.concat([data.iloc[:, 1], data.iloc[:, 2]], ignore_index=True)

score = [0] * len(teamNames)



# teams = [pd.DataFrame({'Name': f"{teamNames}{i}"}) for i in range (30)]
teams = [None] * 30
for i in range (30):
    teams[i] = pd.DataFrame({'Name': teamNames})

print(teams[0])

index  = 0

# # print(df)
for i in range(0, 30):
    teams[i]['goals'] = 0
    teams[i]['Katoxi'] = 0
    teams[i]['totalsouts'] = 0
    teams[i]['target'] = 0
    teams[i]['notarget'] = 0
    teams[i]['corner'] = 0
    teams[i]['apokrousis'] = 0

    # print(teams[i])
    
    data = df.iloc[index:index + 8, :19]
    data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{i + 1}'}, inplace=True)
    data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())



    for j, row in data.iterrows():
        
        # secondTeam = row[3].split('-').str[1].astype(int) + row[4].split('-').str[1].astype(int)
        goals1 = int(row[3].split('-')[0]) + int(row[4].split('-')[0])
        goals2 = int(row[3].split('-')[1]) + int(row[4].split('-')[1])

        if not math.isnan(goals1):
            teams[i].loc[teams[i]['Name'] == row[1], 'goals'] += goals1
        if not math.isnan(goals2):
            teams[i].loc[teams[i]['Name'] == row[2], 'goals'] += goals2

        if not math.isnan(row[7]):
            teams[i].loc[teams[i]['Name'] == row[1], 'Katoxi'] += row[7]
        if not math.isnan(row[8]):
            teams[i].loc[teams[i]['Name'] == row[2], 'Katoxi'] += row[8]

        if not math.isnan(row[9]):
            teams[i].loc[teams[i]['Name'] == row[1], 'totalsouts'] += row[9]
        if not math.isnan(row[10]):    
            teams[i].loc[teams[i]['Name'] == row[2], 'totalsouts'] += row[10]

        if not math.isnan(row[11]):
            teams[i].loc[teams[i]['Name'] == row[1], 'target'] += row[11]
        if not math.isnan(row[12]):
            teams[i].loc[teams[i]['Name'] == row[2], 'target'] += row[12]

        if not math.isnan(row[13]):
            teams[i].loc[teams[i]['Name'] == row[1], 'notarget'] += row[13]
        if not math.isnan(row[14]):
            teams[i].loc[teams[i]['Name'] == row[2], 'notarget'] += row[14]

        if not math.isnan(row[15]):
            teams[i].loc[teams[i]['Name'] == row[1], 'corner'] += row[15]
        if not math.isnan(row[16]):
            teams[i].loc[teams[i]['Name'] == row[2], 'corner'] += row[16]

        if not math.isnan(row[17]):
            teams[i].loc[teams[i]['Name'] == row[1], 'apokrousis'] += row[17]
        if not math.isnan(row[18]):
            teams[i].loc[teams[i]['Name'] == row[2], 'apokrousis'] += row[18]

    index += 9
teamsNp = np.array(teams)
print(teams)
np.save("data.npy", teamsNp)

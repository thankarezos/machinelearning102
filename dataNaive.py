import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import plots as pl

def dataNaive():
    df = pd.read_excel('data.xlsx')

    data = df.iloc[:8, :5]
    data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
    teamNames = pd.concat([data.iloc[:, 1], data.iloc[:, 2]], ignore_index=True)

    score = [0] * len(teamNames)

    teams = pd.DataFrame({'Name': teamNames})

    previousDay = 0
    index  = 0
    for i in range(0, 30):
        data = df.iloc[index:index + 8, :5]
        firstTeam = data.iloc[:, -2].str.split('-').str[0].astype(int) + data.iloc[:, -1].str.split('-').str[0].astype(int)
        secondTeam = data.iloc[:, -2].str.split('-').str[1].astype(int) + data.iloc[:, -1].str.split('-').str[1].astype(int)
        data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{i + 1}'}, inplace=True)
        data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
        data['score1'] = firstTeam
        data['score2'] = secondTeam

        column_name = f'{i + 1}'

        teams[column_name] = previousDay


        for i, row in data.iterrows():
            score1 = row['score1']
            score2 = row['score2']
            if score1 > score2:
                teams.loc[teams['Name'] == row[1], column_name] = 1
                teams.loc[teams['Name'] == row[2], column_name] = 2
            elif score1 < score2:
                teams.loc[teams['Name'] == row[2], column_name] = 1
                teams.loc[teams['Name'] == row[1], column_name] = 2
            else:
                teams.loc[teams['Name'] == row[2], column_name] = 0
                teams.loc[teams['Name'] == row[1], column_name] = 0

        index += 9
    print(teams)
    teams = np.array(teams)[...,1:]
    np.save("nayve.npy", teams)



import pandas as pd

df = pd.read_excel('data.xlsx')

data = df.iloc[:8, :5]
data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
teamNames = pd.concat([data.iloc[:, 1], data.iloc[:, 2]], ignore_index=True)

score = [0] * len(teamNames)

teams = pd.DataFrame({'Name': teamNames, 'Score': score})

previousDay = 0
index  = 0
# # print(df)
for i in range(0, 30):
    data = df.iloc[index:index + 8, :5]
    firstTeam = data.iloc[:, -2].str.split('-').str[0].astype(int) + data.iloc[:, -1].str.split('-').str[0].astype(int)
    secondTeam = data.iloc[:, -2].str.split('-').str[1].astype(int) + data.iloc[:, -1].str.split('-').str[1].astype(int)
    data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{i + 1}'}, inplace=True)
    data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
    # data['score'] = ft.astype(str) + '-' + ht.astype(str)
    data['score1'] = firstTeam
    data['score2'] = secondTeam

    print(data)

    column_name = f'{i + 1}'

    teams[column_name] = previousDay


    for i, row in data.iterrows():
        score1 = row['score1']
        score2 = row['score2']
        if score1 > score2:
            teams.loc[teams['Name'] == row[1], column_name] += 3
            teams.loc[teams['Name'] == row[1], 'Score'] += 3
        elif score1 < score2:
            teams.loc[teams['Name'] == row[2], column_name] += 3
            teams.loc[teams['Name'] == row[2], 'Score'] += 3
        else:
            teams.loc[teams['Name'] == row[1], column_name] += 1
            teams.loc[teams['Name'] == row[2], column_name] += 1
            teams.loc[teams['Name'] == row[1], 'Score'] += 1
            teams.loc[teams['Name'] == row[2], 'Score'] += 1
    previousDay = teams[column_name].copy()

    index += 9
print(teams)


# data = df.iloc[index:index + 8, :5]
# firstTeam = data.iloc[:, -2].str.split('-').str[0].astype(int) + data.iloc[:, -1].str.split('-').str[0].astype(int)
# secondTeam = data.iloc[:, -2].str.split('-').str[1].astype(int) + data.iloc[:, -1].str.split('-').str[1].astype(int)
# data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{0 + 1}'}, inplace=True)
# data.iloc[:, 1:3] = data.iloc[:, 1:3].apply(lambda x: x.str.strip())
# # data['score'] = ft.astype(str) + '-' + ht.astype(str)
# data['score1'] = firstTeam
# data['score2'] = secondTeam

# print(data)






# for index, row in data.iterrows():
#     score1 = row['score1']
#     score2 = row['score2']
#     if score1 > score2:
#         teams.loc[teams['Name'] == row[1], 'Score'] += 3
#     elif score1 < score2:
#         teams.loc[teams['Name'] == row[2], 'Score'] += 3
#     else:
#         teams.loc[teams['Name'] == row[1], 'Score'] += 1
#         teams.loc[teams['Name'] == row[2], 'Score'] += 1

    
    # corresponding_value = row.iloc[1]
    # print(f"Score: {score}, Other Column: {corresponding_value}")


# combined_column = pd.concat([data.iloc[:, 1], data.iloc[:, 2]], ignore_index=True)
# print(combined_column)


# data = data.drop(['FT-Score', 'HT-Score'], axis=1)

# data['New Column2'] = data['New Column'].astype(str) + '-' + 


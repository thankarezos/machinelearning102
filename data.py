import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

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

    # print(data)

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
# teams = teams.iloc[:, ::-1] 
# teams = teams.transpose()



teamsPower = teams.copy()

for i in range(0, 30):
    column_name = f'{i + 1}'
    teamsPower[column_name] = teamsPower[column_name] / ((i+1) * 3)

print(teamsPower)

teamsPowerClasses = teamsPower.copy()

for i in range(0, 30):
    column_name = f'{i + 1}'
    teamsPowerClasses[column_name] = teamsPower[column_name].apply(
        lambda x: 1 if x < 0.1 else 2 if x < 0.3 else 3 if x < 0.5 else 4 if x < 0.7 else 5 if x < 0.9 else 6
    )

print(teamsPowerClasses)


# Get the number of days
num_days = 30

# Create an array of x-axis values for the bars
x = np.arange(num_days)

# Define the width of each bar
bar_width = 0.8

# Define the color for each power class
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate over each day
for day in range(1, num_days + 1):
    column_name = str(day)
    day_data = teamsPowerClasses[column_name].value_counts().sort_index()
    power_classes = np.arange(1, 7)
    day_data = day_data.reindex(power_classes, fill_value=0)

    # Calculate the bottom positions for each power class bar
    bottom = np.zeros(len(power_classes))
    
    # Iterate over each power class
    for j, power_class in enumerate(power_classes):
        counts = day_data.loc[power_class]
        total_teams = 16  # Total number of teams
        
        # Calculate the bar height as a fraction of the total teams
        height = (counts / day_data.sum()) * 16
        
        # Calculate the x-coordinate for the bar
        x_pos = day + j * bar_width / len(power_classes)
        
        # Plot the bar for the power class
        ax.bar(x_pos, height, width=bar_width/len(power_classes), bottom=bottom[j], label=f'Class {power_class}', color=colors[j])
        
        # Update the bottom positions
        bottom[j] += height

# Set the x-axis tick positions and labels
ax.set_xticks(x + 1)
ax.set_xticklabels(range(1, num_days + 1))

# Set the chart title and labels
ax.set_title('Team Power Classes for Each Day')
ax.set_xlabel('Day')
ax.set_ylabel('Number of Teams')

# Add the custom legend
legend_handles = [Patch(facecolor=colors[i], label=f'Class {power_class}') for i, power_class in enumerate(power_classes)]
ax.legend(handles=legend_handles, loc='lower left')

# Adjust the layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the chart
plt.show()

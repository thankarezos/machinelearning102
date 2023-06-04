from matplotlib.patches import Patch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

def stackedBars(teamsPowerClasses, title='Team Power Classes for Each Day'):
    num_days = 30

    # Create an array of x-axis values for the columns
    x = np.arange(num_days)

    # Define the width of each column
    column_width = 0.8

    # Define the color for each power class
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    # Create the stacked column chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)

    # Initialize the bottom positions for each power class
    bottom = np.zeros(num_days)

    # Iterate over each power class
    for power_class in range(1, 7):

        if power_class not in teamsPowerClasses.values:
            continue
        
        class_counts = []
        for day in range(1, num_days + 1):
            column_name = str(day)
            day_data = teamsPowerClasses[column_name].value_counts().sort_index()
            count = day_data.loc[power_class] if power_class in day_data.index else 0
            class_counts.append(count)
        class_counts = np.array(class_counts)
        
        # Plot the stacked column for the power class
        ax.bar(x, class_counts, width=column_width, bottom=bottom, label=f'Class {power_class}', color=colors[power_class-1])
        
        # Update the bottom positions
        bottom += class_counts

    # Set the x-axis tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, num_days + 1))

    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    # Set the chart title and labels
    ax.set_title('Team Power Classes for Each Day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Teams')

    # Add the custom legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=f'Class {i+1}') for i in range(6)]
    ax.legend(handles=legend_handles, loc='lower left')

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the chart
    plt.show()

def barCharts(teamsPowerClasses, title='Team Power Classes for Each Day'):
    num_days = 30

    # Create an array of x-axis values for the bars
    x = np.arange(num_days)

    # Define the width of each bar
    bar_width = 0.8

    # Define the color for each power class
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)

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
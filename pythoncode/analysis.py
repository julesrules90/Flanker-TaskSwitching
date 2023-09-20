import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from config import directory
from datacleaning import save_to_excel

def descriptive_statistics(df, skip_columns=[]):
    stats = {}
    for column in df.columns:
        if column not in skip_columns:
            stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std_dev': df[column].std(),
                'variance':df[column].var(),
                'iqr': df[column].quantile(0.75) - df[column].quantile(0.25),
                'skewness': df[column].skew(),
                'kurtosis': df[column].kurtosis()
            }
    return stats

def calculate_average_per_participant(df):
    cols_to_consider = ['rt_Flanker', 'rt_Task_Switching', 'acc_Flanker', 'acc_Task_Switching']
    average_per_participant = df.groupby('Participant ID')[cols_to_consider].mean()
    return average_per_participant

def histogram(df, unique_id, skip_columns=[]):
    individual_histograms_dir = os.path.join(directory, 'Individual_Histograms')

    for column in df.columns:
        if column not in skip_columns:
            fig, ax = plt.subplots()
            df[column].hist(bins = 100, edgecolor = 'black', ax=ax)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(column)

            fig.savefig(f'{individual_histograms_dir}/histogram_{unique_id}_{column}.png')
            plt.close(fig)

def plot_average_histograms(df):
    df['Participant ID'] = df['Participant ID'].fillna(method='ffill')
    avg_rt_and_acc = df.groupby('Participant ID')[['rt_Flanker', 'rt_Task_Switching', 'acc_Flanker', 'acc_Task_Switching']].mean()

    # Create weights for all the histograms
    weights_rt_Flanker = np.ones_like(avg_rt_and_acc['rt_Flanker']) / len(avg_rt_and_acc['rt_Flanker'])
    weights_rt_Task_Switching = np.ones_like(avg_rt_and_acc['rt_Task_Switching']) / len(avg_rt_and_acc['rt_Task_Switching'])
    weights_acc_Flanker = np.ones_like(avg_rt_and_acc['acc_Flanker']) / len(avg_rt_and_acc['acc_Flanker'])
    weights_acc_Task_Switching = np.ones_like(avg_rt_and_acc['acc_Task_Switching']) / len(avg_rt_and_acc['acc_Task_Switching'])

    # Create the histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].hist(avg_rt_and_acc['rt_Flanker'], bins=8, alpha=0.5, edgecolor='black', weights=weights_rt_Flanker)
    axs[0, 0].set_xlabel('Average RT (Flanker)')
    axs[0, 0].set_ylabel('Percent')
    axs[0, 0].set_title('Average RT per subject for Flanker Task')

    axs[0, 1].hist(avg_rt_and_acc['rt_Task_Switching'], bins=8, alpha=0.5, edgecolor='black', weights=weights_rt_Task_Switching)
    axs[0, 1].set_xlabel('Average RT (Task Switching)')
    axs[0, 1].set_ylabel('Percent')
    axs[0, 1].set_title('Average RT per subject for Task Switching Task')

    axs[1, 0].hist(avg_rt_and_acc['acc_Flanker'], bins=8, alpha=0.5, edgecolor='black', weights=weights_acc_Flanker)
    axs[1, 0].set_xlabel('Average Accuracy (Flanker)')
    axs[1, 0].set_ylabel('Percent')
    axs[1, 0].set_title('Average Accuracy per subject for Flanker Task')

    axs[1, 1].hist(avg_rt_and_acc['acc_Task_Switching'], bins=8, alpha=0.5, edgecolor='black', weights=weights_acc_Task_Switching)
    axs[1, 1].set_xlabel('Average Accuracy (Task Switching)')
    axs[1, 1].set_ylabel('Percent')
    axs[1, 1].set_title('Average Accuracy per subject for Task Switching Task')

    fig.tight_layout()
    plt.savefig(f"{directory}/average_rt_acc_histograms.png")
    plt.close(fig)

def combined_statistics_and_histogram(df):
    skip_columns = ['Participant ID', 'Age', 'Sex', 'Flanker Trial', 'reward_Flanker', 'Task Switching Trial', 'reward_Task_Switching', 'Proportion Congruent', 'Switch Rate']
    stats = descriptive_statistics(df, skip_columns)

    stats_df = pd.DataFrame(stats)
    stats_df.index.name = 'Statistic'
    stats_df.reset_index(inplace=True)

    averages_df = calculate_average_per_participant(df)
    averages_df.reset_index(inplace=True)

    with pd.ExcelWriter(os.path.join(directory, 'Combined_Sample.xlsx'), engine='xlsxwriter') as writer:
        save_to_excel(writer, df, 'Data')
        save_to_excel(writer, stats_df, 'Statistics')
        save_to_excel(writer, averages_df, 'Averages')

    histogram(df, 'Combined', skip_columns)

def plot_avg_rt_by_accuracy(df):
    avg_rt_per_acc_Flanker = df.groupby('acc_Flanker')['rt_Flanker'].mean().reset_index().rename(columns={'acc_Flanker': 'Accuracy', 'rt_Flanker': 'Average RT'})
    avg_rt_per_acc_Flanker['Task'] = 'Flanker'

    avg_rt_per_acc_Task_Switching = df.groupby('acc_Task_Switching')['rt_Task_Switching'].mean().reset_index().rename(columns={'acc_Task_Switching': 'Accuracy', 'rt_Task_Switching': 'Average RT'})
    avg_rt_per_acc_Task_Switching['Task'] = 'Task Switching'

    avg_rt_per_acc = pd.concat([avg_rt_per_acc_Flanker, avg_rt_per_acc_Task_Switching])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.barplot(x='Task', y='Average RT', hue='Accuracy', data=avg_rt_per_acc)

    ax.set_xlabel('Task')
    ax.set_ylabel('Average RT')
    ax.set_title('Average RT for Accurate vs Inaccurate Responses')

    plt.savefig(f'{directory}/average_rt_per_accuracy.png')
    plt.close(fig)

def plot_correlations_from_excel(directory):
    # Load data from Excel
    averages_df = pd.read_excel(directory + 'Combined_Sample.xlsx', sheet_name='Averages')

    # Create 2x1 subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # RT correlation plot
    axs[0].scatter(averages_df['rt_Flanker'], averages_df['rt_Task_Switching'])
    axs[0].set_xlabel('Average RT on Flanker')
    axs[0].set_ylabel('Average RT on Task-Switching')
    axs[0].set_title('Correlation of RT on Flanker vs RT on Task-Switching')

    # Accuracy correlation plot
    axs[1].scatter(averages_df['acc_Flanker'], averages_df['acc_Task_Switching'])
    axs[1].set_xlabel('Average Accuracy on Flanker')
    axs[1].set_ylabel('Average Accuracy on Task-Switching')
    axs[1].set_title('Correlation of Accuracy on Flanker vs Accuracy on Task-Switching')

    # Adjust layout and save the figure
    fig.tight_layout()
    plt.savefig(f'{directory}/correlations.png')
    plt.close(fig)

def plot_avg_rt_by_difficulty_and_reward(df):
    # Filter df to only include correct trials for each task
    flanker_df = df[df['acc_Flanker'] == 1]
    task_switch_df = df[df['acc_Task_Switching'] == 1]

    # Group data by difficulty and reward condition
    flanker_df = flanker_df.groupby(['Proportion Congruent', 'reward_Flanker'])['rt_Flanker'].mean().reset_index()
    task_switch_df = task_switch_df.groupby(['Switch Rate', 'reward_Task_Switching'])['rt_Task_Switching'].mean().reset_index()

    # plot the two tasks
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # plot for Flanker Task
    barplot1 = sns.barplot(x='Proportion Congruent', y='rt_Flanker', hue='reward_Flanker', data=flanker_df, ax=axs[0])
    axs[0].set_title('Average RT by Difficulty and Reward (Flanker Task) for Accurate Trials Only')
    axs[0].set_xlabel('Difficulty Level (Proportion Congruent)')
    axs[0].set_ylabel('Average RT')
    
    # Add labels on top of bars
    for p in barplot1.patches:
        barplot1.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, ha='center', va='bottom')
        
    # plot for Task Switching Task
    barplot2 = sns.barplot(x='Switch Rate', y='rt_Task_Switching', hue='reward_Task_Switching', data=task_switch_df, ax=axs[1])
    axs[1].set_title('Average RT by Difficulty and Reward (Task Switching Task) for Accurate Trials Only')
    axs[1].set_xlabel('Difficulty Level (Switch Rate)')
    axs[1].set_ylabel('Average RT')
    
    # Add labels on top of bars
    for p in barplot2.patches:
        barplot2.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{directory}/average_rt_by_difficulty_and_reward.png')
    plt.close(fig)

def plot_avg_accuracy_by_difficulty_and_reward(df):
    # group data by difficulty and reward condition
    flanker_df = df.groupby(['Proportion Congruent', 'reward_Flanker'])['acc_Flanker'].mean().reset_index()
    task_switch_df = df.groupby(['Switch Rate', 'reward_Task_Switching'])['acc_Task_Switching'].mean().reset_index()

    # plot the two tasks
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # plot for Flanker Task
    barplot1 = sns.barplot(x='Proportion Congruent', y='acc_Flanker', hue='reward_Flanker', data=flanker_df, ax=axs[0])
    axs[0].set_title('Average Accuracy by Difficulty and Reward (Flanker Task)')
    axs[0].set_xlabel('Difficulty Level (Proportion Congruent)')
    axs[0].set_ylabel('Average Accuracy')

    # Add labels on top of bars
    for p in barplot1.patches:
        barplot1.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, ha='center', va='bottom')
    
    # plot for Task Switching Task
    barplot2 = sns.barplot(x='Switch Rate', y='acc_Task_Switching', hue='reward_Task_Switching', data=task_switch_df, ax=axs[1])
    axs[1].set_title('Average Accuracy by Difficulty and Reward (Task Switching Task)')
    axs[1].set_xlabel('Difficulty Level (Switch Rate)')
    axs[1].set_ylabel('Average Accuracy')

    # Add labels on top of bars
    for p in barplot2.patches:
        barplot2.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{directory}/average_accuracy_by_difficulty_and_reward.png')
    plt.close(fig)

def mult_regression(df):
    
    # Change categorical variables to 0 and 1
    df['updated_Proportion_Congruent'] = np.where(df['Proportion Congruent'] == 0.1, 0, np.where(df['Proportion Congruent'] == 0.9, 1, df['Proportion Congruent']))
    df['updated_Switch_Rate'] = np.where(df['Switch Rate'] == 0.1, 0, np.where(df['Switch Rate'] == 0.9, 1, df['Switch Rate']))
        
    df['updated_Reward_Flanker'] = np.where(df['reward_Flanker'] == 1, 0, np.where(df['reward_Flanker'] == 10, 1, df['reward_Flanker']))
    df['updated_Reward_Task_Switching'] = np.where(df['reward_Task_Switching'] == 1, 0, np.where(df['reward_Task_Switching'] == 10, 1, df['reward_Task_Switching']))

    grouped_df = df.groupby(['Participant ID', 'updated_Proportion_Congruent', 'updated_Reward_Flanker'])['rt_Flanker'].mean().reset_index()
    grouped_df_task_switching = df.groupby(['Participant ID', 'updated_Switch_Rate', 'updated_Reward_Task_Switching'])['rt_Task_Switching'].mean().reset_index()

    # Print 
    grouped_df.to_excel(f'{directory}grouped_flanker_data.xlsx', index=False)
    grouped_df_task_switching.to_excel(f'{directory}grouped_task_switching_data.xlsx', index=False)

    # Get Regression
    mod_Flanker = smf.ols("rt_Flanker ~ (updated_Proportion_Congruent) * (updated_Reward_Flanker)", data=grouped_df)
    res_Flanker = mod_Flanker.fit()

    mod_Task_Switching = smf.ols("rt_Task_Switching ~ (updated_Switch_Rate) * (updated_Reward_Task_Switching)", data=grouped_df_task_switching)
    res_Task_Switching = mod_Task_Switching.fit()

    # Get the summary text from the regression result
    summary_text_Flanker = res_Flanker.summary().as_text()
    summary_text_Task_Switching = res_Task_Switching.summary().as_text()

    # Write the summaries to individual text files
    with open(f'{directory}/summary_Flanker.txt', 'w') as file:
        file.write(summary_text_Flanker)
        
    with open(f'{directory}/summary_Task_Switching.txt', 'w') as file:
        file.write(summary_text_Task_Switching)

    print(f"Summaries saved to '{directory}/summary_Flanker.txt' and '{directory}/summary_Task_Switching.txt'")

    return grouped_df, grouped_df_task_switching


def create_bargraphs(grouped_df, grouped_df_task_switching):

    # Calculate the mean for each condition combination
    mean_grouped_df = grouped_df.groupby(['updated_Proportion_Congruent', 'updated_Reward_Flanker'])['rt_Flanker'].mean().reset_index()
    mean_grouped_df_task_switching = grouped_df_task_switching.groupby(['updated_Switch_Rate', 'updated_Reward_Task_Switching'])['rt_Task_Switching'].mean().reset_index()

     # Create a 2-panel figure: 2 rows, 1 column
    plt.figure(figsize=(8, 12))

    # Flanker bar plot
    plt.subplot(2, 1, 1)
    sns.barplot(x='updated_Proportion_Congruent', y='rt_Flanker', hue='updated_Reward_Flanker', data=mean_grouped_df, errorbar=('ci',95))
    plt.xlabel('Updated Proportion Congruent')
    plt.ylabel('Mean RT Flanker')
    plt.title('Flanker')

    # Task Switching bar plot
    plt.subplot(2, 1, 2)
    sns.barplot(x='updated_Switch_Rate', y='rt_Task_Switching', hue='updated_Reward_Task_Switching', data=mean_grouped_df_task_switching, errorbar=('ci',95))
    plt.xlabel('Updated Switch Rate')
    plt.ylabel('Mean RT Task Switching')
    plt.title('Task Switching')

    # Save the entire figure containing both plots
    plt.savefig(f'{directory}/Combined_Regression.png')


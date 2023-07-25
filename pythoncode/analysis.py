import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
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

    if not os.path.exists(individual_histograms_dir):
        os.makedirs(individual_histograms_dir)

    for column in df.columns:
        if column not in skip_columns:
            fig, ax = plt.subplots()
            df[column].hist(bins = 100, edgecolor = 'black', ax=ax)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(column)

            fig.savefig(f'{individual_histograms_dir}/histogram_{unique_id}_{column}.png')
            plt.close(fig)

def plot_average_histograms(combined_df):
    combined_df['Participant ID'] = combined_df['Participant ID'].fillna(method='ffill')
    avg_rt_and_acc = combined_df.groupby('Participant ID')[['rt_Flanker', 'rt_Task_Switching', 'acc_Flanker', 'acc_Task_Switching']].mean()

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
    skip_columns = ['Participant ID', 'Age', 'Sex', 'Flanker Trial', 'reward_Flanker', 'Task Switching Trial', 'reward_Task_Switching']
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
<<<<<<< HEAD:analysis.py
=======

def plot_accuracy_correlation_from_excel(directory):
    # Load data from Excel
    averages_df = pd.read_excel(directory + 'Combined_Sample.xlsx', sheet_name='Averages')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(averages_df['acc_Flanker'], averages_df['acc_Task_Switching'])

    # Add labels and title
    ax.set_xlabel('Average Accuracy on Flanker')
    ax.set_ylabel('Average Accuracy on Task-Switching')
    ax.set_title('Correlation of Accuracy on Flanker vs Accuracy on Task-Switching')

    # Save and close the figure
    plt.savefig(f'{directory}/accuracy_correlation.png')
    plt.close(fig)
>>>>>>> be6bfbcae592bdd84f23569eaf02e35d7f5cd538:pythoncode/analysis.py

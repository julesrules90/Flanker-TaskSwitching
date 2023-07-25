import pandas as pd
import matplotlib.pyplot as plt
import os
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

    # create histogram for average RT (Flanker task)
    fig, ax = plt.subplots(figsize=(10, 6))
    for participant in avg_rt_and_acc.index:
        ax.hist(avg_rt_and_acc.loc[participant, 'rt_Flanker'], bins=30, alpha=0.5, label=f'Participant {participant}')
    ax.set_xlabel('Average RT per Subject (Flanker)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of average RT per subject for Flanker Task')
    ax.legend(loc='upper right')
    plt.savefig(f'{directory}/average_rt_per_subject_histogram_flanker.png')
    plt.close(fig)

    # create histogram for average RT (Task Switching task)
    fig, ax = plt.subplots(figsize=(10, 6))
    for participant in avg_rt_and_acc.index:
        ax.hist(avg_rt_and_acc.loc[participant, 'rt_Task_Switching'], bins=30, alpha=0.5, label=f'Participant {participant}')
    ax.set_xlabel('Average RT per Subject (Task Switching)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of average RT per subject for Task Switching Task')
    ax.legend(loc='upper right')
    plt.savefig(f'{directory}/average_rt_per_subject_histogram_task_switching.png')
    plt.close(fig)

    # create histogram for average Accuracy (Flanker task)
    fig, ax = plt.subplots(figsize=(10, 6))
    for participant in avg_rt_and_acc.index:
        ax.hist(avg_rt_and_acc.loc[participant, 'acc_Flanker'], bins=30, alpha=0.5, label=f'Participant {participant}')
    ax.set_xlabel('Average Accuracy per Subject (Flanker)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of average Accuracy per subject for Flanker Task')
    ax.legend(loc='upper right')
    plt.savefig(f'{directory}/average_acc_per_subject_histogram_flanker.png')
    plt.close(fig)

    # create histogram for average Accuracy (Task Switching task)
    fig, ax = plt.subplots(figsize=(10, 6))
    for participant in avg_rt_and_acc.index:
        ax.hist(avg_rt_and_acc.loc[participant, 'acc_Task_Switching'], bins=30, alpha=0.5, label=f'Participant {participant}')
    ax.set_xlabel('Average Accuracy per Subject (Task Switching)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of average Accuracy per subject for Task Switching Task')
    ax.legend(loc='upper right')
    plt.savefig(f'{directory}/average_acc_per_subject_histogram_task_switching.png')
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

def plot_rt_correlation_from_excel(directory):
    # Load data from Excel
    averages_df = pd.read_excel(directory + 'Combined_Sample.xlsx', sheet_name='Averages')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(averages_df['rt_Flanker'], averages_df['rt_Task_Switching'])

    # Add labels and title
    ax.set_xlabel('Average RT on Flanker')
    ax.set_ylabel('Average RT on Task-Switching')
    ax.set_title('Correlation of RT on Flanker vs RT on Task-Switching')

    # Save and close the figure
    plt.savefig(f'{directory}/rt_correlation.png')
    plt.close(fig)

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

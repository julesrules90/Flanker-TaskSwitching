import pandas as pd
import matplotlib.pyplot as plt
from datacleaning import main, save_to_excel
from config import directory

def descriptive_statistics(df, skip_columns=[]):
    stats = {}
    for column in df.columns:
        if column not in skip_columns:
            stats[column] = {
                "mean": df[column].mean(),
                "median": df[column].median(),
                "std_dev": df[column].std(),
                "variance":df[column].var(),
                "iqr": df[column].quantile(0.75) - df[column].quantile(0.25),
                "skewness": df[column].skew(),
                "kurtosis": df[column].kurtosis()
            }
    return stats

def histogram(df, directory, unique_id, skip_columns=[]):
    for column in df.columns:
        if column not in skip_columns:
            fig, ax = plt.subplots()
            df[column].hist(bins = 100, edgecolor = "black", ax=ax)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(column)
            fig.savefig(f"{directory}/histogram_{unique_id}_{column}.png")
            plt.close(fig)

if __name__ == "__main__":
    dataframes = main()  # Calls the main function from datacleaning.py and gets the return data
    for i, df in enumerate(dataframes):

        unique_id = int(df.loc[0, 'Participant ID'])
        with pd.ExcelWriter(directory + f'combined_{unique_id}.xlsx', engine='xlsxwriter') as writer:

            save_to_excel(writer, df, 'Data')

            skip_columns = ["Participant ID", "Age", "Sex", "Flanker Trial", "reward_Flanker", "Task Switching Trial", "reward_Task_Switching"]
            stats = descriptive_statistics(df, skip_columns)

            stats_df = pd.DataFrame(stats)
            stats_df.index.name = 'Statistic'
            stats_df.reset_index(inplace=True)
            
            save_to_excel(writer, stats_df, 'Statistics')

        histogram(df, directory, unique_id, skip_columns)
        print(f"Finished statistics and histograms for participant {unique_id}")
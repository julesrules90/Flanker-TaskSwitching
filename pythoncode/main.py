import pandas as pd
import os
from config import directory
from datacleaning import main, save_to_excel
from analysis import *

if __name__ == '__main__':
    dataframes, combined_df = main()

    skip_columns = ['Participant ID', 'Age', 'Sex', 'Flanker Trial', 'reward_Flanker', 'Task Switching Trial', 'reward_Task_Switching']

    for i, df in enumerate(dataframes):
        unique_id = int(df.loc[0, 'Participant ID'])
        with pd.ExcelWriter(os.path.join(directory, f'combined_{unique_id}.xlsx'), engine='xlsxwriter') as writer:
            save_to_excel(writer, df, 'Data')

            stats = descriptive_statistics(df, skip_columns)
            stats_df = pd.DataFrame(stats)
            stats_df.index.name = 'Statistic'
            stats_df.reset_index(inplace=True)

            save_to_excel(writer, stats_df, 'Statistics')

        histogram(df, unique_id, skip_columns)
        print(f'Finished statistics and histograms for participant {unique_id}')

    plot_average_histograms(combined_df)

    combined_statistics_and_histogram(combined_df)
    print('Finished creating Combined_Sample.xlsx')

    plot_avg_rt_by_accuracy(combined_df)

    plot_correlations_from_excel(directory)
    print('Finished Scatterplots')

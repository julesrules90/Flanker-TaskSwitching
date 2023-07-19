import glob
import pandas as pd
import numpy as np
import os
from config import directory, Flanker_Folder, Task_Switching_Folder, get_columns_Flanker, get_columns_Task_Switching

def get_file_paths(directory, folder, unique_id):
    path = os.path.join(directory, folder, f"*{unique_id}*.csv")
    return glob.glob(path)

def get_participant_info(file_path):
    df_info = pd.read_csv(file_path, nrows=1)
    return df_info["age"][0], df_info["sex"][0]

def create_info_df(unique_id, participant_age, participant_sex):
    return pd.DataFrame({
                "Participant ID": [unique_id],
                "Age": [participant_age],
                "Sex": [participant_sex],
            })

def read_and_prepare_new_df(file_path, columns, task_name, unique_id):
    df = pd.read_csv(file_path, usecols=columns)
    # adds task trial number in new dataframe
    if 'acc' in df.columns and df['acc'].count() > 0:
        df.insert(0, f'{task_name} Trial', np.where(df['acc'].isnull(), np.nan, range(1, df['acc'].count()+1)))
    else:
        print(f"No 'acc' column or empty 'acc' column in {task_name} file for unique_id {unique_id}")
    return df

def save_to_excel(writer, df, sheet_name):
    df.to_excel(writer, index=False, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]

    # formats the columns to make the excel file more readable
    for idx, col in enumerate(df): 
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),
            len(str(series.name))  
            )) + 1  
        worksheet.set_column(idx, idx, max_len) 

def main():

    dataframes = []
    for unique_id in range(10001, 10016): 
        csv_files_Flanker = get_file_paths(directory, Flanker_Folder, unique_id)
        csv_files_Task_Switching = get_file_paths(directory, Task_Switching_Folder, unique_id)

        if csv_files_Flanker and csv_files_Task_Switching:
            print(f"For unique_id {unique_id}, found matching files for flanker and task switching")
            participant_age, participant_sex = get_participant_info(csv_files_Flanker[0])

            df_info = create_info_df(unique_id, participant_age, participant_sex)
            
            df1 = read_and_prepare_new_df(csv_files_Flanker[0], get_columns_Flanker, 'Flanker', unique_id)
            df1 = df1.rename(columns={"rt": "rt_Flanker", "reward": "reward_Flanker", "acc": "acc_Flanker"})

            df2 = read_and_prepare_new_df(csv_files_Task_Switching[0], get_columns_Task_Switching, 'Task Switching', unique_id)
            df2 = df2.rename(columns={"rt": "rt_Task_Switching", "reward": "reward_Task_Switching", "acc": "acc_Task_Switching"}) 

            df = pd.concat([df_info, df1, df2], axis=1)

            dataframes.append(df)
            
    return dataframes
    
if __name__ == "__main__":
    main()
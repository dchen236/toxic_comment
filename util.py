from config import sample_submission_csv as SAMPLE_CSV_PATH
import pandas as pd
import numpy as np

def save_submission(predictions, save_name):
    submission = pd.read_csv(SAMPLE_CSV_PATH, index_col='id')
    submission['prediction'] = predictions
    submission.to_csv(save_name)
    print("saved prediction as: %s" %save_name)


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df, toxic_col, identity_columns):
    bool_df = df.copy()
    for col in [toxic_col] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df


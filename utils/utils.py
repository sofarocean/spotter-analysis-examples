import os
import re
from datetime import datetime
import time
import numpy as np
import pandas as pd
import math

def find_files_with_string(directory, string, case_sensitive=False):
    """
    Returns a list of filenames in the given directory that contain the given string.

    Args:
        directory (str): Path to the directory to search.
        string (str): String to search for.
        case_sensitive (bool): Whether to match case-sensitively. Default is False.

    Returns:
        List[str]: List of matching filenames.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"'{directory}' is not a valid directory.")

    matches = []
    for filename in os.listdir(directory):
        if not case_sensitive:
            if string in filename.lower():
                matches.append(filename)
        else:
            if string in filename:
                matches.append(filename)

    sorted_files = sorted(matches, key=lambda x: int(re.search(r'\d+', x).group()))

    return sorted_files

def epoch_to_date_array( epochtime ):

    datetime = np.array([
        list(time.gmtime(x))[0:6] if not (isinstance(x, float) and math.isnan(x)) else [np.nan] * 6
        for x in epochtime
    ])
    milis = np.array([
        1000 * (x - np.floor(x)) if not (isinstance(x, float) and math.isnan(x)) else np.nan
        for x in epochtime
    ])
    return np.concatenate((datetime, milis[:, None]), axis=1)


def format_csv_time(data, keep_duplicates=False):

    columns = list(data.columns)

    time_tuple = data[columns[0:7]].values
    time = []
    for index in range(time_tuple.shape[0]):
        try:
            time.append(
                datetime(
                    year=time_tuple[index, 0].astype(int),
                    month=time_tuple[index, 1].astype(int),
                    day=time_tuple[index, 2].astype(int),
                    hour=time_tuple[index, 3].astype(int),
                    minute=time_tuple[index, 4].astype(int),
                    second=time_tuple[index, 5].astype(int),
                    microsecond=1000*np.round(time_tuple[index, 6]).astype(int)
                )
            )
        except ValueError:
            time.append(np.nan)

    if not keep_duplicates:
        time, unique = np.unique(np.array(time), return_index=True)
    else:
        time = np.array(time)
    return time

def read_spotter_csv(filename):

    data = pd.read_csv(filename)
    new_df = pd.DataFrame()
    new_df['time'] = format_csv_time(data)
    ncols = len(data.columns)-7
    new_df = pd.concat([new_df, data.iloc[:, -ncols:]], axis=1)

    return new_df

def generate_date_pairs(start_date, end_date, freq='2W', precision='D'):

    # Convert to pandas Timestamp and generate date range with 2-week frequency
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Ensure last date is included if necessary
    if dates[-1] < pd.to_datetime(end_date):
        dates = dates.append(pd.DatetimeIndex([pd.to_datetime(end_date)]))

    # Create tuples of 2-week periods
    date_pairs = [
        (np.datetime64(dates[i].to_pydatetime(), precision),
         np.datetime64(dates[i+1].to_pydatetime(), precision))
        for i in range(len(dates)-1)
    ]

    return date_pairs
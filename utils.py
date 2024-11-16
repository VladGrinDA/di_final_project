import pandas as pd

def data_info(df):
    df.info()
    return ('Describe numeric:', df.describe(),
            'Describe object:', df.describe(include='object'),
            'Nulls:', df.isnull().sum())
    
    
def remove_empty_columns(df):
    null_columns = df.columns[df.isnull().sum() == len(df)]
    return df.drop(columns=null_columns), null_columns


import json
import pandas as pd

def replace_single_quotes(df, columns):
    """
    Replace single quotes with double quotes in specified columns.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing JSON columns with single quotes.
    - columns (list): List of column names to replace single quotes in.
    
    Returns:
    - pd.DataFrame: DataFrame with quotes replaced in specified columns.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: x.replace("'", '"') if pd.notnull(x) else x)
    return df

def process_retention_graphs(df, retention_column):
    """
    Process retention graph columns to extract metrics.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing retention graph columns.
    - retention_column (str): The column name of the retention graph JSON.
    
    Returns:
    - pd.DataFrame: DataFrame with new columns for retention metrics.
    """
    # Convert JSON strings to dictionaries
    retention_data = df[retention_column].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
    
    # Calculate Average Retention Rate
    df[f'{retention_column}_avg'] = retention_data.apply(lambda x: sum(x.values()) / len(x) if x else None)
    
    # Calculate Initial Retention Drop (first interval vs middle interval)
    df[f'{retention_column}_initial_drop'] = retention_data.apply(
        lambda x: (list(x.values())[0] - list(x.values())[len(x) // 2]) / list(x.values())[0] if x else None
    )
    
    # Calculate Retention Consistency Score (standard deviation)
    df[f'{retention_column}_consistency'] = retention_data.apply(
        lambda x: pd.Series(list(x.values())).std() if x else None
    )
    
    def avg_drop_rate(retention_dict):
        if not retention_dict:
            return None
        values = list(retention_dict.values())
        drops = [(values[i] - values[i + 1]) / values[i] if values[i] != 0 else 0 for i in range(len(values) - 1)]
        return sum(drops) / len(drops) if drops else 0

    df[f'{retention_column}_avg_drop_rate'] = retention_data.apply(avg_drop_rate)
    
    return df


def expand_json_columns(df, json_columns):
    """
    Expands JSON columns into separate columns in the given DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing JSON format columns.
    - json_columns (list): List of columns with JSON data.
    
    Returns:
    - pd.DataFrame: Expanded DataFrame with JSON columns in relational format.
    """
    for col in json_columns:
        # Parse JSON data
        expanded = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
        # Normalize and add prefix with original column name
        expanded_df = pd.json_normalize(expanded)
        expanded_df.columns = [f"{col}_{sub_col}" for sub_col in expanded_df.columns]
        # Concatenate the expanded columns to the original DataFrame
        df = pd.concat([df, expanded_df], axis=1)
        # Drop the original JSON column
        df.drop(columns=[col], inplace=True)
        
    return df

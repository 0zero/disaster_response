import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

NUMBER_OF_CATEGORIES = 36


def load_data(messages_filepath, categories_filepath):
    """
    Load in data from file paths and merge them together into a single dataframe

    :param messages_filepath: path to messages.csv
    :param categories_filepath: path to categories.csv
    :return: dataframe with combined data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on="id")
    categories = categories.categories.str.split(";", expand=True)

    column_names = categories.iloc[0].apply(lambda x: x[:-2])
    categories.columns = column_names

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def get_nan_columns(df):
    """
     Return the column names where all their data is NaN from a dataframe
    :param df:
    :return: list of column names where all their data is NaN
    """
    nan_cols = []
    for col in df.columns:
        if df[col].isna().all():
            nan_cols.append(col)
    return nan_cols


def clean_data(df):
    """
    Cleans data from given dataframe. Remove duplicates, drop some NaNs
    :param df: data frame from load_data function
    :return: same dataframe but with duplicates and NaNs removed
    """
    if df.duplicated(keep="first").sum() > 0:
        df.drop_duplicates(inplace=True)

    if len(get_nan_columns(df)) != 0:
        df.dropna(axis=1, subset=get_nan_columns(df), inplace=True)

    # Categories are our target attributes so we don't want any missing data in these. Since all the category
    # columns got created from a single one, they'll all have the same missing data so we can just look at one and
    # remove everything
    if np.sum(df.related.isna()) > 0:
        cols = df.columns[len(df.columns)-NUMBER_OF_CATEGORIES:]
        df.dropna(axis=0, subset=cols, inplace=True)

    # The related category has 3 options: 0,1,2 but we are supposed to have a binary setup
    # We can either remove rows with related==2 or convert 2 -> 1. Ideally, we'd want to run
    # both scenarios to check that there isn't a significant effect of doing either. Also, rows
    # with related==2 are zero for every other category so I think it's safe to convert 2->1.
    df[df.loc[:, "related"] == 2] = 1

    return df


def save_data(df, database_filename):
    """
    Export data to sqlite database
    :param df:
    :param database_filename:
    :return: None
    """
    # TODO: add try-except statement here
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('DisasterTweets', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print("Please provide the filepaths of the messages and categories "
              "datasets as the first and second argument respectively, as "
              "well as the filepath of the database to save the cleaned data "
              "to as the third argument. \n\nExample: python process_data.py "
              "disaster_messages.csv disaster_categories.csv "
              "DisasterResponse.db")


if __name__ == "__main__":
    main()

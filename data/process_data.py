import sys
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the messages and categories datasets and merges them
    
    Parameters:
    messages_filepath - Messages csv filepath
    categories_filepath - Categories csv filepath
    
    Output:
    merged_df - Merged messages and categories dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    merged_df = messages.merge(categories)

    return merged_df


def clean_data(df):
    '''
    Explode category column into seperate columns. Remove duplicates.
    
    Parameters:
    df - input df (merged)
    
    Output:
    clean_df - cleaned df
    '''
    
    #explode categories column into many
    categories = df['categories'].str.split(';', expand=True)
    
    #get category names, assign as column names
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
      
    categories.columns = category_colnames
    
    #remove category names from values, convert to int type
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    #related column has value 2, transform to 1
    categories['related'] = categories['related'].map(lambda x: 1 if x==2 else x)

    #replace categories column with the new exploded categories df
    df.drop(['categories'], axis=1, inplace=True)
    
    clean_df = pd.concat([df, categories], axis=1)
    
    #remove duplicates
    clean_df.drop_duplicates(inplace=True)

    return clean_df


def save_data(df, database_filename):
    '''
    Saves data to a sqlite database
    
    Parameters:
    df - cleaned_df
    database_filename - intended filename/filepath for db
    '''
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False,         if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
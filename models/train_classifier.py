import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
import pickle

def load_data(database_filepath):
    '''
    Loads data from database, splits into X (features) and Y (target) dataframes. Returns X, Y, list of unique Y categories
    
    Parameters:
    database_filepath - filepath for db file
    
    Output:
    X - features
    Y - target variable
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = list(set(Y))
   
    return X, Y, categories

def tokenize(text):
    '''
    Converts input to text to lowercase. Tokenizes and lemmatizes.
    
    Parameters:
    Text - text input
    
    Output:
    final_tokens - tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    final_tokens=[]
    for token in tokens:
        final_tokens.append(lemmatizer.lemmatize(token).lower().strip())
        
    return final_tokens

def build_model():
    '''
    Builds model from ML pipeline and performs hyperparameter tuning based on designated hyperparameters

    Output:
    model - final tuned model
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    
    parameters = {
    'clf__estimator__n_estimators': [50, 100]
}
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv = 2, verbose = 3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates performance of model. Returns accuracy level for each column.
    
    Parameters:
    model - chosen classifier
    X_test - test features
    Y_test - test target variable
    
    Returns:
    Accuracy for each column
    '''
    
    Y_pred = model.predict(X_test)
    (Y_pred == Y_test).mean()

def save_model(model, model_filepath):
    '''
    Save model as .pkl file to intended filepath.
    
    Parameters:
    model - classificatin model
    model_filepath - intended filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        nltk.download(['punkt', 'wordnet'])
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
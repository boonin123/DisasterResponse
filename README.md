# Disaster Response Pipeline Project
This project's purpose is to develop both ML and ETL pipelines for real messages sent during disaster events. The pipelines will then be used in a web app so emergency workers can input new messages and receive classification results. 

### ETL Pipeline
The ETL pipeline merges the disaster_messages.csv and disaster_categories.csv datasets, cleans them, and adds them into a Database file DisasterResponse.db

### ML Pipeline
The ML pipelines takes the Disaster Response database as input. It then trains and tunes a decision tree classifier and saves it as the file classifier.pkl

### Files
APP
templates - templates for web app
run.py - python file to run web app

DATA
disaster_messages.csv - input dataset
disaster_categories.csv - input dataset
DisasterResponse.db - target Database file
process_data.py - python script to carry out ETL Pipeline (see above)

MODELS
train_classifier.py - python script to carry out ML Pipeline (see above)
classifier.pkl (not included) - target model file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

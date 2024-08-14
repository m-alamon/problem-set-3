'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Load the CSV files
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Extract genres from genres_df
    genre_list = genres_df['genre'].unique().tolist()
    
    # Initialize dictionaries for counts
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    # Process model predictions
    for _, row in model_pred_df.iterrows():
        imdb_id = row['imdb_id']
        predicted_genres = row['predicted'].split(',')
        actual_genres = ast.literal_eval(row['actual genres'])
        correct = row['correct?']
        
        # Count true genres
        for genre in actual_genres:
            if genre:  # Skip empty genres
                genre_true_counts[genre] += 1
        
        # Count true positives and false positives
        for genre in predicted_genres:
            if genre:
                if genre in actual_genres:
                    if correct == 1:
                        genre_tp_counts[genre] += 1
                else:
                    if correct == 0:
                        genre_fp_counts[genre] += 1
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts

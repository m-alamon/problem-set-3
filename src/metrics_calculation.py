'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Initialize dictionaries for tp, fp, tn, fn
    tp = {genre: 0 for genre in genre_list}
    fp = {genre: 0 for genre in genre_list}
    fn = {genre: 0 for genre in genre_list}
    tn = {genre: 0 for genre in genre_list}

    # Calculate tp, fp, fn, tn for each genre
    for genre in genre_list:
        tp[genre] = genre_tp_counts.get(genre, 0)
        fp[genre] = genre_fp_counts.get(genre, 0)
        fn[genre] = genre_true_counts.get(genre, 0) - tp[genre]
        tn[genre] = len(model_pred_df) - (tp[genre] + fp[genre] + fn[genre])

    # Calculate precision, recall, and F1 score for each genre
    precision = {genre: tp[genre] / (tp[genre] + fp[genre]) if (tp[genre] + fp[genre]) > 0 else 0 for genre in genre_list}
    recall = {genre: tp[genre] / (tp[genre] + fn[genre]) if (tp[genre] + fn[genre]) > 0 else 0 for genre in genre_list}
    f1_score = {genre: 2 * (precision[genre] * recall[genre]) / (precision[genre] + recall[genre]) if (precision[genre] + recall[genre]) > 0 else 0 for genre in genre_list}

    # Calculate micro metrics
    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Calculate macro metrics
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1_score = np.mean(list(f1_score.values()))

    # Return values in the expected format
    return micro_precision, micro_recall, micro_f1_score, list(precision.values()), list(recall.values()), list(f1_score.values())

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    pred_rows = []
    true_rows = []

    # Populate pred_rows and true_rows
    for _, row in model_pred_df.iterrows():
        predicted_genres = row['predicted'].split(',')
        actual_genres = ast.literal_eval(row['actual genres'])
        pred_row = [1 if genre in predicted_genres else 0 for genre in genre_list]
        true_row = [1 if genre in actual_genres else 0 for genre in genre_list]
        pred_rows.append(pred_row)
        true_rows.append(true_row)

    # Convert to DataFrames
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)

    # Calculate metrics using sklearn
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)

    return precision, recall, f1_score, micro_precision, micro_recall, micro_f1_score

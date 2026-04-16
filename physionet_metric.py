import numpy as np
import pandas as pd

def calculate_physionet_utility(df_results, id_col="Patient_ID", label_col="label", pred_col="prediction"):
    """
    Calculates the official PhysioNet 2019 Normalized Utility Score.
    """
    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3
    max_u_tp   = 1.0   
    min_u_fn   = -2.0  
    u_fp       = -0.05 

    total_score = 0.0
    total_optimal = 0.0
    total_no_pred = 0.0

    for pid, group in df_results.groupby(id_col):
        labels = group[label_col].values
        preds = group[pred_col].values
        n = len(labels)

        t_sepsis = np.argmax(labels) + 6 if np.any(labels) else None
        u_true = np.zeros(n)

        if t_sepsis is not None:
            for t in range(n):
                dt = t - t_sepsis
                if dt <= dt_early:
                    u_true[t] = u_fp
                elif dt <= dt_optimal:
                    u_true[t] = u_fp + (max_u_tp - u_fp) * (dt - dt_early) / (dt_optimal - dt_early)
                elif dt <= dt_late:
                    u_true[t] = max_u_tp + (min_u_fn - max_u_tp) * (dt - dt_optimal) / (dt_late - dt_optimal)
                else:
                    u_true[t] = min_u_fn
        else:
            u_true[:] = u_fp

        score = 0
        for t in range(n):
            if preds[t] == 1:
                score += u_true[t]
            elif t_sepsis is not None and t >= t_sepsis + dt_late:
                score += min_u_fn

        optimal_preds = (u_true > 0).astype(int)
        optimal_score = 0
        for t in range(n):
            if optimal_preds[t] == 1:
                optimal_score += u_true[t]
            elif t_sepsis is not None and t >= t_sepsis + dt_late:
                optimal_score += min_u_fn

        no_pred_score = 0
        if t_sepsis is not None:
            no_pred_score = min_u_fn * max(0, n - (t_sepsis + dt_late))

        total_score += score
        total_optimal += optimal_score
        total_no_pred += no_pred_score

    if total_optimal == total_no_pred:
        return 0.0
        
    unormalized = (total_score - total_no_pred) / (total_optimal - total_no_pred)
    return unormalized
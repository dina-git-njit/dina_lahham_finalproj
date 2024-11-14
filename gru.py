import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, roc_curve, auc

def gru(X, y, kf):
    X_reshaped = np.array(X).astype('float32').reshape((X.shape[0], X.shape[1], 1))
    
    fold_metrics = {
        'TP': [], 'TN': [], 'FP': [], 'FN': [], 'P': [], 'N': [],
        'TPR': [], 'TNR': [], 'FPR': [], 'FNR': [],
        'Recall': [], 'Precision': [], 'F1 Score': [],
        'Accuracy': [], 'Error Rate': [], 'BACC': [],
        'TSS': [], 'HSS': [],
        'Brier Score': [], 'AUC': []
    }

    tpr_list = []
    fpr_list = []
    auc_scores = [] 

    cumulative_TP, cumulative_TN, cumulative_FP, cumulative_FN = 0, 0, 0, 0

    for train_index, test_index in kf.split(X_reshaped, y):
        X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            GRU(256, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0022692999466626704),
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        
        cumulative_TN += TN
        cumulative_FP += FP
        cumulative_FN += FN
        cumulative_TP += TP

        P = TP + FN
        N = TN + FP 
        
        TPR = TP / P if P != 0 else 0  # Sensitivity/Recall
        TNR = TN / N if N != 0 else 0  # Specificity
        FPR = FP / N if N != 0 else 0
        FNR = FN / P if P != 0 else 0 
        recall = TPR
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (TP + TN) / (P + N)
        error_rate = 1 - accuracy
        brier = brier_score_loss(y_test, y_pred_prob)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        bacc = (TPR + TNR) / 2
        tss = TPR - FPR 
        hss = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) != 0 else 0
        
        fold_metrics['TP'].append(TP)
        fold_metrics['TN'].append(TN)
        fold_metrics['FP'].append(FP)
        fold_metrics['FN'].append(FN)
        fold_metrics['P'].append(P)
        fold_metrics['N'].append(N)
        fold_metrics['TPR'].append(TPR)
        fold_metrics['TNR'].append(TNR)
        fold_metrics['FPR'].append(FPR)
        fold_metrics['FNR'].append(FNR)
        fold_metrics['Recall'].append(recall)
        fold_metrics['Precision'].append(precision)
        fold_metrics['F1 Score'].append(f1)
        fold_metrics['Accuracy'].append(accuracy)
        fold_metrics['Error Rate'].append(error_rate)
        fold_metrics['BACC'].append(bacc)
        fold_metrics['TSS'].append(tss)
        fold_metrics['HSS'].append(hss)
        fold_metrics['Brier Score'].append(brier)
        fold_metrics['AUC'].append(auc_score)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_scores.append(auc_score)

    sum_metrics = ['TP', 'TN', 'FP', 'FN', 'P', 'N']
    mean_metrics = [metric for metric in fold_metrics if metric not in sum_metrics]
    
    sum_values = {metric: [np.sum(fold_metrics[metric])] for metric in sum_metrics}
    mean_values = {metric: [np.mean(fold_metrics[metric])] for metric in mean_metrics}

    sum_df = pd.DataFrame(sum_values).T
    sum_df.columns = ['Sum']
    mean_df = pd.DataFrame(mean_values).T
    mean_df.columns = ['Mean']

    fold_metrics_df = pd.concat([sum_df, mean_df, pd.DataFrame(fold_metrics).T], axis=1)
    fold_metrics_df.columns = ['Sum', 'Mean'] + [f'{i+1}' for i in range(10)]

    print(fold_metrics_df)

    cumulative_cm = np.array([[cumulative_TP, cumulative_FP],
                              [cumulative_FN, cumulative_TN]])
    print("\nCumulative Confusion Matrix:")
    print(cumulative_cm)

    plt.figure(figsize=(10, 8))
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], linestyle='--', label=f'Fold {i+1} AUC = {auc_scores[i]:.2f}')

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(len(tpr_list))], axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Non-Discrimination')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - GRU Model - 10-Fold Cross-Validation')
    plt.legend(loc='lower right')
    plt.show()

    return fold_metrics_df

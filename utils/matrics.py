import ipdb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
                "regression": self.__eval_regression,
                "classification": self.__eval_classification,
            }
        
    def __eval_classification(self, y_pred, y_true):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        # six classes
        y_pred_6 = np.argmax(y_pred, axis=1)
        Mult_acc = accuracy_score(y_pred_6, y_true)
        F1_score = f1_score(y_true, y_pred_6, average='macro')

        eval_results = {
            "acc": round(Mult_acc, 4),
            "f1_score": round(F1_score, 4)
        }
        return eval_results
    
    def __eval_regression(self, y_pred, y_true):
        y_pred = y_pred.view(-1).cpu().detach().numpy()
        y_true = y_true.view(-1).cpu().detach().numpy()

        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)   # Average L1 distance between preds and truths
        rmse = np.sqrt(np.mean(np.square(y_pred - y_true))).astype(np.float64)
        corr = np.corrcoef(y_pred, y_true)[0][1]
        
        eval_results = {
            "corr": round(corr, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }
        return eval_results

    def getMetics(self, train_mode):
        return self.metrics_dict[train_mode]
"""Models and search spaces of the hyperparameter optimisation experiment.

This module is kept free of GPU imports because cv_error is executed in
worker processes.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
except Exception:
    xgb = None

# Every hyperparameter is encoded in [0, 1]; the table lists the decoded ranges.
SEARCH_SPACE = {
    'SVM_RBF': [('C', 'log10-uniform', '1e-2', '1e3'), ('gamma', 'log10-uniform', '1e-4', '1e1')],
    'XGBoost': [('n_estimators', 'int', '50', '400'), ('max_depth', 'int', '2', '10'),
                ('learning_rate', 'log10-uniform', '1e-2', '10^-0.3 (0.501)'), ('subsample', 'uniform', '0.5', '1.0'),
                ('colsample_bytree', 'uniform', '0.5', '1.0'), ('min_child_weight', 'uniform', '1', '10'),
                ('gamma', 'uniform', '0', '5'), ('reg_lambda', 'log10-uniform', '1e-3', '10')],
}


def map_xgb(v):
    return dict(n_estimators=int(50 + v[0] * 350),
                max_depth=int(2 + v[1] * 8),
                learning_rate=float(10 ** (-2 + v[2] * 1.7)),
                subsample=float(0.5 + 0.5 * v[3]),
                colsample_bytree=float(0.5 + 0.5 * v[4]),
                min_child_weight=float(1 + 9 * v[5]),
                gamma=float(5 * v[6]),
                reg_lambda=float(10 ** (-3 + 4 * v[7])))


def build_model(kind, v):
    """Classifier for a point v of the unit cube."""
    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    if kind == 'SVM_RBF':
        return make_pipeline(StandardScaler(), SVC(C=10 ** (-2 + 5 * v[0]), gamma=10 ** (-4 + 5 * v[1])))
    return xgb.XGBClassifier(**map_xgb(v), tree_method='hist', n_jobs=1, verbosity=0, random_state=0)


def cv_error(kind, v, Xtr, ytr, inner_seed):
    """Three-fold cross-validation error on the training part (1 if the fit fails)."""
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(inner_seed))
        return 1.0 - float(cross_val_score(build_model(kind, v), Xtr, ytr, cv=cv, scoring='accuracy').mean())
    except Exception:
        return 1.0

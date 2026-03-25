import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_candidate_classifiers(random_state=42):
    return {
        'logreg': LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'),
        'rf': RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced'),
        'xgb': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
        'lgbm': LGBMClassifier(random_state=random_state)
    }


def evaluate_model(X, y, clf, cv=5, scoring=None):
    if scoring is None:
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    res = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False)
    return res

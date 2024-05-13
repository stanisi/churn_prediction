import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_classifier(df, feature_names, target_name):

    classifier = XGBClassifier(eval_metric = 'auc', max_depth = 4, n_estimators = 250)

    features = df[feature_names]
    target = df[target_name]

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size = 0.85)

    classifier.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = False)
    return classifier

def features_to_prediction(daily_df, feature_names, classifier) -> dict:
    
    features_df = daily_df[feature_names] # chore: i don't like this

    preds = classifier.predict_proba(features_df)
    preds_fed = pd.Series(preds, index = daily_df['FED_ID']) # FED_ID is a unique player identifier
    preds_fed_min = preds_fed.min(level='FED_ID')

    return preds.to_dict(), preds_fed_min.to_dict()
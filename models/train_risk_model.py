import pickle
import pandas as pd 
import xgboost as xgb
from pathlib import Path
from preprocessing.feature_engineering import load_dataset, preprocess, get_features_targets
from configparser import ConfigParser 
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of models/
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

DATA_PATH = BASE_DIR / config["data_path"]

df = load_dataset(DATA_PATH)
df = preprocess(df)
X, y = get_features_targets(df)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=5,
    n_estimators=300,
    learning_rate=0.05
)
model.fit(X, y)

with open(config["model_path"], "wb") as f:
    pickle.dump(model, f)

print("XGBoost model trained and saved at", config["model_path"])


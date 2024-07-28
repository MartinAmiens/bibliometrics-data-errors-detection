import neptune
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Initialize a Neptune run
run = neptune.init_run(
    project="martin-amiens/HAL-data-names-errors-findind",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDZiMTJjYy02MmEyLTRiNGUtODc5Ny1lNDgxM2EwOTY2YjcifQ"
)

# Load a dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define and log model parameters
params = {"n_estimators": 100, "max_depth": 2, "random_state": 42}
run["parameters"] = params

# Train a simple model
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Log training and evaluation metrics
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_f1 = f1_score(y_train, train_preds, average="macro")
test_f1 = f1_score(y_test, test_preds, average="macro")

run["train/f1_score"] = train_f1
run["eval/f1_score"] = test_f1

# Stop the run
run.stop()

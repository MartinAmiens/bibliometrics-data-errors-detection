import neptune

# Initialize a Neptune run
run = neptune.init_run(
    project="martin-amiens/HAL-data-names-errors-findind",
    api_token="your_api_token_here"
)

# Log some dummy parameters
params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

# Log some dummy training loss values
for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

# Log a dummy evaluation metric
run["eval/f1_score"] = 0.66

# Stop the run
run.stop()

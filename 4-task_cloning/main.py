from clearml import Task
import time

# Initialize ClearML Task
task = Task.init(project_name="demo", task_name="hyperparameter_example")

# Define hyperparameters
args = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10, "optimizer": "adam", "dropout": 0.2}

# Connect hyperparameters to ClearML (editable in UI)
args = task.connect(args)

# Clear visual demonstration of hyperparameter effects
print("=" * 50)
print("HYPERPARAMETERS RECEIVED:")
print("=" * 50)
for key, value in args.items():
    print(f"{key}: {value}")
print("=" * 50)

# Simulate training with visible hyperparameter impact
for epoch in range(args["epochs"]):
    simulated_loss = 1.0 / (epoch + 1) * (1 / args["learning_rate"])
    print(
        f"Epoch {epoch + 1}/{args['epochs']} | Batch Size: {args['batch_size']} | Loss: {simulated_loss:.4f} | Optimizer: {args['optimizer']}"
    )
    time.sleep(0.5)

print(f"\nTraining complete with dropout={args['dropout']}")

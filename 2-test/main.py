from clearml import Task

# Create a test task (this will appear in the ClearML Web UI)
task = Task.init(project_name="demo", task_name="test_run")

# Log a simple scalar
for i in range(5):
    task.get_logger().report_scalar("test_metric", "iteration", value=i, iteration=i)

print("✅ ClearML test complete! Check your ClearML Web UI.")

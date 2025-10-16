import clearml
from clearml import Task, Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Initialize ClearML task for resuming
task = Task.init(project_name="demo", task_name="checkpoint_resume_demo")


# Simple model (must match original architecture)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Get previous task's checkpoint
# Method 1: Using task ID (replace with your actual task ID)
previous_task_id = "8363af96d4f8417583834335112ec9dd"  # Get this from ClearML UI
previous_task = Task.get_task(task_id=previous_task_id)

# Download the last checkpoint artifact
checkpoint_path = previous_task.artifacts["checkpoint_epoch_9.pth"].get_local_copy()

# Alternative Method 2: Using output model
# model_id = 'YOUR_MODEL_ID'  # Get from ClearML UI
# input_model = Model(model_id)
# checkpoint_path = input_model.get_local_copy()

# Alternative Method 3: Query for latest task
# tasks = Task.get_tasks(project_name='demo', task_name='checkpoint_demo')
# if tasks:
#     previous_task = tasks[0]
#     checkpoint_path = previous_task.artifacts['checkpoint_epoch_9.pth'].get_local_copy()

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
print(f"Resuming from epoch {checkpoint['epoch'] + 1}, Loss: {checkpoint['loss']:.4f}")

# Initialize model and optimizer
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load saved states
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
best_loss = checkpoint["loss"]

# Create dummy dataset (same as before for demo)
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()

# Continue training from checkpoint
for epoch in range(start_epoch, start_epoch + 10):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Log to ClearML
    task.get_logger().report_scalar("loss", "train", avg_loss, epoch)

    # Save new checkpoints
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, f"best_checkpoint_resumed.pth")
        task.upload_artifact("best_checkpoint", artifact_object=f"best_checkpoint_resumed.pth")
        print(f"New best model saved with loss: {best_loss:.4f}")

# Final model save
torch.save(model.state_dict(), "final_model_resumed.pth")
task.update_output_model(model_path="final_model_resumed.pth")

print("Resumed training complete")
task.close()

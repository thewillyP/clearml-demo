import clearml
from clearml import Task
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Initialize ClearML task
task = Task.init(project_name="demo", task_name="resume_from_checkpoint")

# Get previous task's output model
previous_task = Task.get_task(task_id="c8321a734d2a483584a673c38dbf0be1")
checkpoint_path = previous_task.models["output"][-1].get_local_copy()


# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Create dummy dataset
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
print(f"Resuming from epoch {start_epoch}, previous loss: {checkpoint['loss']:.4f}")

# Continue training
for epoch in range(start_epoch, start_epoch + 5):
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
    task.get_logger().report_scalar("loss", "train", avg_loss, epoch)

print("Training complete")
task.close()

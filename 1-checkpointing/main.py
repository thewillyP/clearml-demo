import clearml
from clearml import Task
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Initialize ClearML task
task = Task.init(project_name="demo", task_name="checkpoint_demo")


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

# Training loop with checkpointing
for epoch in range(10):
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

    # Save checkpoint - ClearML automatically captures this
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    torch.save(checkpoint, f"1-checkpointing/checkpoint_epoch_{epoch}.pth")

    # Register as model artifact
    if epoch % 3 == 0:  # Save every 3 epochs as output model
        task.update_output_model(model_path=f"1-checkpointing/checkpoint_epoch_{epoch}.pth")

print("Training complete")
task.close()

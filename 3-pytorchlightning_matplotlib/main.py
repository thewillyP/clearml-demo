import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for ClearML
import matplotlib.pyplot as plt

from clearml import Task

# Initialize ClearML FIRST - enables automatic logging
task = Task.init(project_name="demo", task_name="auto_logging")


class SimpleNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 128)
        self.l2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return self.l2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# Load MNIST
dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST("", train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])


# Setup TensorBoard logger - ClearML auto-captures these logs
tb_logger = TensorBoardLogger("tb_logs", name="mnist")

# Train - ClearML auto-captures all metrics
model = SimpleNet()
trainer = pl.Trainer(max_epochs=2, logger=tb_logger, log_every_n_steps=10)

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

trainer.fit(model, train_loader, val_loader)


# Matplotlib plot - ClearML auto-captures this
plt.figure(figsize=(10, 5))
for i in range(8):
    img, label = dataset[i]
    plt.subplot(2, 4, i + 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
plt.suptitle("MNIST Samples")
plt.tight_layout()
plt.show()
plt.close()


# Another matplotlib plot
plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3], [0.5, 0.3, 0.1], "b-", linewidth=2)
plt.title("Fake Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
plt.close()

print("\n✅ ClearML automatically captured:")
print("- Matplotlib plots (check Debug Samples tab)")
print("- TensorBoard scalars (train_loss, val_loss)")
print("- Model architecture")
print("- Hyperparameters")
print("- Console output")

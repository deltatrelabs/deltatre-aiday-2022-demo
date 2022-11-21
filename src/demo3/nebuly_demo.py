from pathlib import Path
import torch
import torchvision.models as models
from nebullvm.api.functions import optimize_model

print(f"Cuda: {torch.cuda.is_available()}")

# Load a resnet as example
model = models.resnet50()

# Provide input data for the model    
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0])) for _ in range(100)]

# Run nebullvm optimization
optimized_model = optimize_model(model, input_data=input_data,  optimization_time="unconstrained")

# Save model
save_path = Path("artifacts")
Path.mkdir(save_path, exist_ok=True)

optimized_model.save(save_path)

import torch
from PIL import Image
import numpy as np
from transformers import ViTModel, ViTImageProcessor
from torchvision.transforms import Resize, ToTensor

# Specify the model name or identifier from the Hugging Face model hub
model_name = "./vit-bs32-lr2em5"

# Load the VIT model
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Load the VIT model
model = ViTModel.from_pretrained(model_name)

# model = torch.load('vit-bs32-lr2em5/pytorch_model.bin')
image_path = "./test.jpeg"
image = Image.open(image_path).convert("RGB")
resized_image = Resize((224, 224))(image)
input_tensor = ToTensor()(resized_image).unsqueeze(0)

# Set the model to evaluation mode
model.eval()

# Export the model to ONNX
torch.onnx.export(model, input_tensor, "vit_model.onnx", opset_version=11)

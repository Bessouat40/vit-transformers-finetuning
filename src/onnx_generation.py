import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

model_name = "../vit-bs32-lr2em5"

model = ViTForImageClassification.from_pretrained(model_name)

image_processor = ViTImageProcessor.from_pretrained(model_name)

image_path = "./test.jpeg"
image = Image.open(image_path).convert("RGB")

inputs = image_processor(images=image, return_tensors="pt")

input_tensor = inputs["pixel_values"]

model.eval()

torch.onnx.export(
    model=model,
    args=(input_tensor,),
    f="vit_model2.onnx",
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

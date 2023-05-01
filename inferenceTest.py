from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained("BobCalifornia/v1-vit-pneumonia")
model = AutoModelForImageClassification.from_pretrained("BobCalifornia/v1-vit-pneumonia")

image = Image.open("./test.jpeg")

encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

with torch.no_grad():
  outputs = model(**encoding)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

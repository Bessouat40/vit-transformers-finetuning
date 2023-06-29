from transformers import pipeline
from PIL import Image

image = Image.open("./test.jpeg").convert("RGB")
image2 = Image.open("./test-copy.jpeg").convert("RGB")
print(image)
print(image2)
images = [image, image2]

model_name = 'BobCalifornia/v1-vit-pneumonia'
model = pipeline(model=model_name, tokenizer=model_name)

outputs = model(images)

predictions = []

for output in outputs :
    if output[0]['score'] > output[1]['score'] :
      predictions.append('0')
    else : predictions.append('1')

print(predictions)

import onnxruntime
from PIL import Image
import numpy as np

ort_sess = onnxruntime.InferenceSession("./vit_model.onnx",
                                        providers=['CPUExecutionProvider'])
                                        # providers=['CUDAExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

img_paths = ["./test.jpeg", "./test.jpeg"]

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    img = np.array(image, dtype=np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

inputs = np.array([load_img(path) for path in img_paths])
outputs = ort_sess.run([output_name], {input_name: inputs})[0]

logits = np.array(outputs)

probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

predicted_class = np.argmax(probabilities, axis=1)

print("Predicted classes :", predicted_class)
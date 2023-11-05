import onnxruntime
from PIL import Image
import numpy as np
import time

ort_sess = onnxruntime.InferenceSession("./vit_model2.onnx",
                                        providers=['CUDAExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

times = []

for i in range(10):
    image = Image.open("./test.jpeg").convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    img = np.array(image, dtype=np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    inputs = np.expand_dims(img, axis=0)
    st = time.time()
    outputs = ort_sess.run([output_name], {input_name: inputs})[0]

    logits = np.array(outputs)

    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    predicted_class = np.argmax(probabilities, axis=1)

    print("Probabilités:", probabilities)
    print("Classe prédite:", predicted_class)

    times.append(time.time() - st)

print(f"ONNX time performance: {np.mean(times):.4f} seconds")

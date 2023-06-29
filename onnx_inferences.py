import onnxruntime
from PIL import Image
import numpy as np
import time

ort_sess = onnxruntime.InferenceSession("vit_model.onnx",
                                        providers = ['CUDAExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

### ONNX model performance with CUDA provider
times = []
for i in range(10):
    image = Image.open("./test.jpeg").convert("RGB")
    img = np.array(image, dtype=np.float32)
    img_shape = (224,224,3)
    img_width, img_height, nb_canaux = img_shape[0], img_shape[1], img_shape[2]
    inputs = np.array([np.resize(img,(nb_canaux, img_width, img_height))])
    st = time.time()
    outputs = ort_sess.run([output_name], {input_name : inputs})[0]
    print(outputs)
    times.append(time.time()-st)
print(f"ONNX time performance: {np.mean(times):.4f}")
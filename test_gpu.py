from model import keras_lstm
import numpy as np

import onnxmltools
import onnxruntime
from keras.models import load_model

seq_length = 30
dat_size = 10
dat_dim = 30
fake_data = np.random.random(size=(dat_size,seq_length,dat_dim)).astype(np.float32)
gpu_model_keras = keras_lstm(data_dim=dat_dim,timesteps=seq_length,gpu=True)
cpu_model_keras = keras_lstm(data_dim=dat_dim,timesteps=seq_length,gpu=False)

output_onnx_model_cpu = 'model_cpu.onnx'
output_onnx_model_gpu = 'model_gpu.onnx'

# Convert the Keras model into ONNX
onnx_model_cpu = onnxmltools.convert_keras(cpu_model_keras)
onnx_model_gpu = onnxmltools.convert_keras(gpu_model_keras)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model_cpu, output_onnx_model_cpu)
onnxmltools.utils.save_model(onnx_model_gpu, output_onnx_model_gpu)

sess = onnxruntime.InferenceSession(output_onnx_model_cpu)

pred_onnx_cpu = sess.run(None, {'lstm_1_input':fake_data})
pred_keras_cpu = cpu_model_keras.predict(fake_data)

print('onnx cpu output: ', pred_onnx_cpu)
print('keras output: ', pred_keras_cpu)
print('diff onnx cpu: ', pred_keras_cpu - pred_onnx_cpu)


sess = onnxruntime.InferenceSession()

pred_onnx_gpu = sess.run(None, {'lstm_1_input':fake_data})
pred_keras_gpu = gpu_model_keras.predict(fake_data)

print('onnx gpu output: ', pred_onnx_gpu)
print('keras output: ', pred_keras_gpu)
print('diff onnx gpu: ', pred_keras_gpu - pred_onnx_gpu)

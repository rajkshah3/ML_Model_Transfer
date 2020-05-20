#https://github.com/onnx/keras-onnx
from model import keras_lstm, keras_lstm_bidirectional
import numpy as np

import onnxmltools
import onnxruntime
from keras.models import load_model

seq_length = 30
dat_size = 10
dat_dim = 30
fake_data = np.random.random(size=(dat_size,seq_length,dat_dim)).astype(np.float32)
# gpu_model_keras = keras_lstm(data_dim=dat_dim,timesteps=seq_length,gpu=True)
cpu_model_keras = keras_lstm_bidirectional(data_dim=dat_dim,timesteps=seq_length,gpu=False)

output_onnx_model_cpu = 'model_cpu.onnx'
output_onnx_model_gpu = 'model_gpu.onnx'

# Convert the Keras model into ONNX
onnx_model_cpu = onnxmltools.convert_keras(cpu_model_keras)
# onnx_model_gpu = onnxmltools.convert_keras(gpu_model_keras)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model_cpu, output_onnx_model_cpu)
# onnxmltools.utils.save_model(onnx_model_gpu, output_onnx_model_gpu)

sess = onnxruntime.InferenceSession(output_onnx_model_cpu)
fake_data = fake_data if(isinstance(fake_data,list)) else [fake_data]
feed = dict([(inp.name, fake_data[n]) for n, inp in enumerate(sess.get_inputs())])

pred_onnx = sess.run(None, feed)
pred_keras = cpu_model_keras.predict(fake_data)

print('onnx output: ', pred_onnx)
print('keras output: ', pred_keras)
print('diff: ', pred_keras - pred_onnx)
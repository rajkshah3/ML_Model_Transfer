from model import keras_lstm, keras_lstm_bidirectional
import numpy as np

from keras.models import load_model

seq_length = 30
dat_size = 10
dat_dim = 30
fake_data = np.random.random(size=(dat_size,seq_length,dat_dim)).astype(np.float32)
gpu_model_keras = keras_lstm_bidirectional(data_dim=dat_dim,timesteps=seq_length,gpu=True)
cpu_model_keras = keras_lstm_bidirectional(data_dim=dat_dim,timesteps=seq_length,gpu=False)

output_keras_model_cpu = 'model_cpu_weights.h5'
output_keras_model_gpu = 'model_gpu_weights.h5'

# Convert the Keras model into ONNX
cpu_model_keras.save_weights(output_keras_model_cpu)
gpu_model_keras.save_weights(output_keras_model_gpu)


cpu_model_keras.load_weights(output_keras_model_gpu)

reloaded_gpu_model = cpu_model_keras

reloaded_output = reloaded_gpu_model.predict(fake_data)
gpu_model_output = gpu_model_keras.predict(fake_data)

print('gpu output: ', gpu_model_output)
print('reloaded output: ', reloaded_output)
print('diff reloaded cpu: ', gpu_model_output - reloaded_output)


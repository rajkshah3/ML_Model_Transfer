import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,Flatten,Bidirectional

def keras_lstm(data_dim=100,timesteps=30,units=128,gpu=True):
    from keras.layers import LSTM
    model_cpu = Sequential()
    #For CudnnLSTM compatibility
    #activations of LSTM fixed (tanh and sigmoid)
    model_cpu.add(LSTM(units,activation='tanh',recurrent_activation='sigmoid', return_sequences=True, stateful=False,input_shape=(timesteps,data_dim,)))
    model_cpu.add(LSTM(int(units/4),activation='tanh',recurrent_activation='sigmoid',return_sequences=False))
    # model_cpu.add(keras.layers.Flatten())
    model_cpu.add(Dropout(0.3))
    model_cpu.add(Dense(units=int(units/16)))
    model_cpu.add(Dense(units=1, activation='sigmoid'))
    model_cpu.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])
    model_cpu.summary()
    model = model_cpu

    if(gpu):
        from keras.layers import CuDNNLSTM as LSTM
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, stateful=False,input_shape=(timesteps,data_dim)))
        model.add(LSTM(int(units/4), return_sequences=False, stateful=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=int(units/16)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])
        model.summary()

    return model


def keras_lstm_bidirectional(data_dim=100,timesteps=30,units=128,gpu=True):
    from keras.layers import LSTM

    model_cpu = Sequential()
    #For CudnnLSTM compatibility
    #activations of LSTM fixed (tanh and sigmoid)
    model_cpu.add(Bidirectional(LSTM(units,activation='tanh',recurrent_activation='sigmoid', return_sequences=True, stateful=False),input_shape=(timesteps,data_dim,)))
    model_cpu.add(Bidirectional(LSTM(int(units/4),activation='tanh',recurrent_activation='sigmoid',return_sequences=False)))
    # model_cpu.add(keras.layers.Flatten())
    model_cpu.add(Dropout(0.3))
    model_cpu.add(Dense(units=int(units/16)))
    model_cpu.add(Dense(units=1, activation='sigmoid'))
    model_cpu.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])
    model_cpu.summary()
    model = model_cpu

    if(gpu):
        from keras.layers import CuDNNLSTM as LSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(units, return_sequences=True, stateful=False),input_shape=(timesteps,data_dim,)))
        model.add(Bidirectional(LSTM(int(units/4),return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(units=int(units/16)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])
        model.summary()

    return model

def keras_gru_bidirectional(data_dim=100,timesteps=30,units=128,gpu=True):
    from keras.layers import GRU

    model_cpu = Sequential()
    # For CudnnGRU compatibility:
    # recurrent activations gru must be fixed to sigmoid
    # reset_after must be set to true 
    model_cpu.add(Bidirectional(GRU(units,recurrent_activation='sigmoid',reset_after=True, return_sequences=True, stateful=False),input_shape=(timesteps,data_dim,)))
    model_cpu.add(Bidirectional(GRU(int(units/4),recurrent_activation='sigmoid',return_sequences=False,reset_after=True)))
    # model_cpu.add(keras.layers.Flatten())
    model_cpu.add(Dropout(0.3))
    model_cpu.add(Dense(units=int(units/16)))
    model_cpu.add(Dense(units=1, activation='sigmoid'))
    model_cpu.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])
    model_cpu.summary()
    model = model_cpu

    if(gpu):
        from keras.layers import CuDNNGRU as GRU
        model = Sequential()
        model.add(Bidirectional(GRU(units, return_sequences=True, stateful=False),input_shape=(timesteps,data_dim,)))
        model.add(Bidirectional(GRU(int(units/4),return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(units=int(units/16)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])
        model.summary()

    return model
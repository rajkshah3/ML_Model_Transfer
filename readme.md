<!-- ABOUT THE PROJECT -->
## Keras to Onnx model conversion


Repository used to test Keras and tensorflow model conversion libraries for serving.
Precicion of test cases were used for each model. 

Type of models
* Recurrent
* Convolutional
* GPU/CPU

* Keras to onnx
* TF GPU to CPU
* Unidirectional and Bidirectional

### Built With

* Keras
* Tensorflow
* Onnx
* TF Serving


## Findings

### It Keras to onnx model transfer was found to work in all cases

### It is possible to convert the GPU models to CPU under certain conditions:

   * Model weights from the GPU models are dumped out to disk and then read in by the CPU models
   * The CPU LSTM and GRU models need specific setup for this to work
   * Certain features of the CPU LSTM/GRU modules are not available in the final model because of the limitations of the GPU model
       + Recurrent activations are fixed in the GPU implementations
       + Internal biases in the GPU are coupled whereas they are separate in the CPU implementation
   * CPU LSTM and GRU models can subsequently be exported to an universal (onnx) format for serving with alternative frameworks
   * Inference outputs were tested on untrained models and found differences of the order of 10e-8
   * The tested models, chosen for similarity to POC model, composed of:
       + Bidirectional sequence layer (GRU and LSTM)
       + Bidirectional sequence layer (GRU and LSTM)
       + Dense layer
       + Single output node in final layer

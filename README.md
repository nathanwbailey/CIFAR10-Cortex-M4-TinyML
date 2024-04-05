# CIFAR-10 CNN on Arduino Nano 33 BLE Sense Rev 2

### What is this project?

This project implements a lightweight CNN for CIFAR-10 based on linearwise blocks from the EtinyNet network (https://ojs.aaai.org/index.php/AAAI/article/view/20387). 

This is implemented in Keras and then converted to TFLite. This is deployed using TfLite-Micro on an Arduino Nano 33 BLE Sense Rev 2 which has a Cortex-M4F Microcontroller.

### Blogs

In addition to the code I wrote a blog on this project:

### Where is the code?

The code is located in the following files:

* micro_controller_neural_net.py - Implements the CNN using linearwise blocks in Keras, converts it to tflite and outputs an image in a C header.
* micro_controller_neural_net_dense_linear_blocks.py - Implements the CNN using dense linearwise blocks in Keras, converts it to tflite and outputs an image in a C header.
* cortex_m4_program/cortex_m4_program.ino - Runs the CNN on the Cortex-M4F, classifies the example outputted in the python file.

### Requirements

All pip packages needed can be found in requirements.txt

### How to Run

1. Run the python file: e.g. python3 micro_controller_neural_net.py
2. Convert the tflite model to a C header:
    * apt-get install xxd
    * xxd -i cifar10.tflite > model.h 
    * sed -i 's/unsigned char/const unsigned char/g' model.h
    * sed -i 's/const/alignas(8) const/g' model.h
3. Run the arduino C file

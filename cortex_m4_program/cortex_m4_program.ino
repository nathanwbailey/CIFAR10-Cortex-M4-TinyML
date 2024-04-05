// Include the tensorflow micro libraries
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h> 
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

//include the model and the input
#include "model_linear_blocks.h"
#include "input_linear_blocks.h"
//Global variables for the model, I/O, arena size, interpreter and quantization parameters
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

//RAM usage of the model should be about 36KB
//Set Tensor Arena Size to 44KB
//This is done by looking at the Tensor Arena Size after creating the interpreter
//Additionally this can be found out using Edge Impluse
constexpr int tensor_arena_size = 44000;
uint8_t* tensor_arena;

//Quantization parameters
float o_scale = 0.0f;
int32_t o_zero_point = 0;

void setup() {
  //Init Serial
  Serial.begin(115200);
  while (!Serial);
  //Allocate the tensor_arena on the heap, aligned on a 16-byte boundary
  tensor_arena = new __attribute__((aligned(16))) uint8_t[tensor_arena_size];
  //Load the Model
  model = tflite::GetModel(cifar_classifier_tflite);

  // MicroMutableOpResolver is used by the interpreter to register and access the operations that are used by the model
  // This is used in contrast to AllOpsResolver which registers all the DNN operations supported by tflite-micro
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddMean();

  //Create the interpreter
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    tensor_arena_size
  );
  interpreter = &static_interpreter;
  //Allocate Tensors
  //This Runs through the model and allocates all necessary input, output and intermediate tensors
  interpreter->AllocateTensors();
  //Get the input and the output TFLite Tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  //Find the actual tensor arena size, this is fed back to the tensor_arena_size
  Serial.println(interpreter->arena_used_bytes());

  //Get the quantization parameters
  //This is contained in TfLiteAffineQuantization struct
  const auto *o_quant = reinterpret_cast<TfLiteAffineQuantization*>(output->quantization.params);
  //o_scale and o_zero_point are found in looking at the needed arrays
  o_scale = o_quant->scale->data[0];
  o_zero_point = o_quant->zero_point->data[0];

}

void loop() {
  //Copy our input image into the input tensor
  std::memcpy(tflite::GetTensorData<int8_t>(input), g_test, g_test_len);
  //Invoke the interpreter
  interpreter->Invoke();

  int32_t ix_max = 0;
	float pb_max = 0;
  //Get the output tensor, loop through the outputs
  //Get the value of each output, dequantize it to find the probability
  //Max probability index gives our predicted class which we print out
  int8_t* out_val = tflite::GetTensorData<int8_t>(output);
	for (int32_t ix = 0; ix <= 10; ix++) {
    int8_t o_val = out_val[ix];
		float pb = ((float) o_val-o_zero_point) * o_scale;
		if (pb > pb_max) {
			ix_max = ix;
			pb_max = pb;
		}
	}
  Serial.println(ix_max);
  //Add a loop so we do this once
  while(1);
}

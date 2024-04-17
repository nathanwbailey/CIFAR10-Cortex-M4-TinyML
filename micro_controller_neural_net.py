"""Neural Network Created and Quantized for tflite-micro."""
from typing import Generator
from pathlib import Path
import os
import time
import tensorflow as tf # type: ignore[import-untyped]
from tensorflow import keras # type: ignore[reportAttributeAccessIssue,import-untyped] # pylint: disable=no-member,import-error,no-name-in-module
import numpy as np
import pandas as pd # type: ignore[import-untyped]
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ------------------------------------- #
# -- Create the Classification Model -- #
# ------------------------------------- #
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
val_images = val_images / 255.0

class LinearBottleneckBlock(keras.layers.Layer):
    """Custom Linear Bottleneck Layer from EtinyNet."""
    def __init__(self, out_channels: int, kernel_size: int, padding: str = 'same', strides: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.depthwise_conv_layer_a = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, strides=strides, use_bias=bias)
        self.depthwise_a_batch_norm_layer = keras.layers.BatchNormalization()

        self.pointwise_layer = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', strides=1, use_bias=bias)
        self.pointwise_batch_norm = keras.layers.BatchNormalization()

        self.depthwise_conv_layer_b = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", strides=1, use_bias=bias)
        self.depthwise_b_batch_norm_layer = keras.layers.BatchNormalization()

        self.activation = keras.layers.Activation('relu')

    def call(self, input_tensor: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward Pass for the Linear Bottleneck Layer."""
        depthwise_result = self.depthwise_a_batch_norm_layer(self.depthwise_conv_layer_a(input_tensor), training=training)
        pointwise_result = self.activation(self.pointwise_batch_norm(self.pointwise_layer(depthwise_result), training=training))
        output = self.activation(self.depthwise_b_batch_norm_layer(self.depthwise_conv_layer_b(pointwise_result), training=training))
        return output

model = keras.Sequential([
    keras.layers.Conv2D(filters = 16, kernel_size=5, padding='same', strides=1, input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    LinearBottleneckBlock(out_channels=32, kernel_size=3),
    LinearBottleneckBlock(out_channels=64, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    LinearBottleneckBlock(out_channels=64, kernel_size=3),
    LinearBottleneckBlock(out_channels=128, kernel_size=3),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=10, kernel_regularizer=keras.regularizers.L2(1e-3))
])

model.summary(expand_nested=True)

loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
LEARNING_RATE = 0.01
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, min_lr=0, min_delta=0.001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=8, min_delta=0.001)

logging_directory_name = "tensorboard_log_dir"
if not Path(logging_directory_name).exists():
    Path(logging_directory_name).mkdir()
root_logdir = os.path.join(os.curdir, logging_directory_name)

def get_run_logdir(root_logdir_in: str) -> str:
    """Return a folder for the run to use in Tensorboard."""
    run_id = time.strftime("linear_run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir_in, run_id)

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir(root_logdir))

model.fit(
    train_images,
    train_labels,
    epochs=100,
    batch_size=32,
    verbose=2,
    validation_data=(val_images, val_labels),
    callbacks = [lr_scheduler, early_stopping, tensorboard_cb]
)
model.save('cifar_classifier')

# ----------------------- #
# -- Convert to TFLite -- #
# ----------------------- #
cifar_ds = tf.data.Dataset.from_tensor_slices(train_images)
def representative_dataset_function() -> Generator[list, None, None]:
    """Create a representative dataset for TFLite Conversion."""
    for input_value in cifar_ds.batch(1).take(100):
        i_value_fp32 = tf.cast(input_value, tf.float32)
        yield [i_value_fp32]

converter = tf.lite.TFLiteConverter.from_saved_model('cifar_classifier')
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_function)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore[reportAttributeAccessIssue]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8 # type: ignore[reportAttributeAccessIssue]
converter.inference_output_type = tf.int8 # type: ignore[reportAttributeAccessIssue]


tflite_model = converter.convert()
with open("cifar_classifier.tflite", "wb") as f:
    f.write(tflite_model) # type: ignore[reportAttributeAccessIssue]

tflite_model_kb_size = os.path.getsize("cifar_classifier.tflite") / 1024
print(tflite_model_kb_size)

tflite_interpreter = tf.lite.Interpreter(model_content = tflite_model)
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()[0]
output_details = tflite_interpreter.get_output_details()[0]

input_quantization_details = input_details["quantization_parameters"]
output_quantization_details = output_details["quantization_parameters"]
input_quant_scale = input_quantization_details['scales'][0]
output_quant_scale = output_quantization_details['scales'][0]
input_quant_zero_point = input_quantization_details['zero_points'][0]
output_quant_zero_point = output_quantization_details['zero_points'][0]


def classify_sample_tflite(interpreter: tf.lite.Interpreter, input_d: dict, output_d: dict, i_scale: np.float32, o_scale: np.float32, i_zero_point: np.int32, o_zero_point: np.int32, input_data: np.ndarray) -> tf.Tensor:
    """Classify an example in TFLite."""
    input_data = input_data.reshape((1,32,32,3))
    input_fp32 = tf.cast(input_data, tf.float32)
    input_int8 = tf.cast((input_fp32 / i_scale) + i_zero_point, tf.int8)
    interpreter.set_tensor(input_d["index"], input_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_d["index"])[0]
    output_fp32 = tf.convert_to_tensor((output_int8 - o_zero_point) * o_scale, dtype=tf.float32)
    return output_fp32

num_correct_examples = 0
for i_value, o_value in zip(val_images, val_labels):
    output = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)
    if np.argmax(output) == o_value:
        num_correct_examples += 1

print(f'Accuracy: {num_correct_examples/len(list(val_images))}')


def array_to_str(data: np.ndarray) -> str:
    """Convert numpy array of int8 values to comma seperated int values."""
    num_cols = 10
    val_string = ''
    for i, val in enumerate(data):
        val_string += str(val)
        if (i+1) < len(data):
            val_string += ','
        if (i+1) % num_cols == 0:
            val_string += '\n'
    return val_string

def generate_h_file(size: int, data: str, label: str) -> str:
    """Generate a c header with the string numpy data."""
    str_out = 'int8_t g_test[] = '
    str_out += '\n{\n'
    str_out += f'{data}'
    str_out += '};\n'
    str_out += f'const int g_test_len = {size};\n'
    str_out += f'const int g_test_label = {label};\n'
    return str_out

imgs = list(zip(val_images, val_labels))
cols = ["Image", "Label"]

df = pd.DataFrame(imgs, columns=cols)

frog_samples = df[df['Label'] == 6]
c_code = ""
for index, row in frog_samples.iterrows():
    i_value = np.asarray(row['Image'].tolist(), dtype=np.float32)
    o_value = np.asarray(row['Label'].tolist(), dtype=np.float32)
    o_pred_fp32 = classify_sample_tflite(tflite_interpreter, input_details, output_details, input_quant_scale, output_quant_scale, input_quant_zero_point, output_quant_zero_point, i_value)

    if np.argmax(o_pred_fp32) == o_value:
        i_value_int8 = ((i_value / input_quant_scale) + input_quant_zero_point).astype(np.int8)
        i_value_int8 = i_value_int8.ravel()

        val_str = array_to_str(i_value_int8)
        c_code = generate_h_file(i_value_int8.size, val_str, "6")

with open('input_linear_blocks.h', 'w', encoding='utf-8') as file:
    file.write(c_code)

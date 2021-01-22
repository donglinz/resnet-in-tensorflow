import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--deterministic_init', action='store_true')
parser.add_argument('--deterministic_input', action='store_true')
parser.add_argument('--deterministic_tf', action='store_true')
parser.add_argument('--ckpt_folder', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--save_ckpt', action='store_true')
parser.add_argument('--save_pred', action='store_true')
args = parser.parse_args()

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import resnet50
from tensorflow.keras.initializers import GlorotUniform

import os
if args.deterministic_tf:
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import itertools

if not os.path.exists(args.ckpt_folder):
    os.makedirs(args.ckpt_folder)
  
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
  

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = args.batch_size
IMG_SHAPE = 32

trainloader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess_image(image, label):
  img = tf.cast(image, tf.float32)
  img = img/255.

  return img, label
if args.deterministic_input:
    trainloader = (
        trainloader
        .shuffle(1024, seed=0)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
else:
    trainloader = (
        trainloader
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

testloader = (
    testloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

def Model():
  inputs = keras.layers.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3))
  if args.deterministic_init:
    initializer = GlorotUniform(seed=0)
  else:
    initializer = GlorotUniform()

  x = keras.layers.Conv2D(16, (3,3), padding='same', kernel_initializer=initializer)(inputs)
  # x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.Conv2D(32,(3,3), padding='same', kernel_initializer=initializer)(x)
  # x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.Conv2D(32,(3,3), padding='same', kernel_initializer=initializer)(x)
  # x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(32, activation='relu', kernel_initializer=initializer)(x)
  # x = keras.layers.Dropout(0.1)(x)
  
  outputs = keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(x)

  return keras.models.Model(inputs=inputs, outputs=outputs)


tf.keras.backend.clear_session()
model = Model()

tf.keras.utils.plot_model(
    model, to_file='small_cnn.png', show_shapes=True, show_layer_names=True, dpi=65
)

# Custom LR schedule as mentioned in the LossLandscape paper
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 1.6*1e-3),
    (9, (1.6*1e-3)/2),
    (19, (1.6*1e-3)/4),
    (29, (1.6*1e-3)/8),
]

def lr_schedule(epoch):
    if (epoch >= 0) & (epoch < 9):
        return LR_SCHEDULE[0][1]
    elif (epoch >= 9) & (epoch < 19):
        return LR_SCHEDULE[1][1]
    elif (epoch >= 19) & (epoch < 29):
        return LR_SCHEDULE[2][1]
    else:
        return LR_SCHEDULE[3][1]

import hashlib
def get_weight_hash():
  with open(os.path.join(args.ckpt_folder, 'hash.txt'), 'a') as f:
    for w in model.weights:
      name = w.name
      w = w.numpy()
      w.flags.writeable = False
      
      f.write(name + ' ' + hashlib.md5(w.tobytes()).hexdigest() + '\n')
def get_input_hash():
  arr_x = []
  arr_y = []
  for x, y in trainloader:
    arr_x.extend(np.split(x.numpy(), x.numpy().shape[0], axis=0))
    arr_y.extend(np.split(y.numpy(), y.numpy().shape[0], axis=0))
  for x, y in trainloader:
    arr_x.extend(np.split(x.numpy(), x.numpy().shape[0], axis=0))
    arr_y.extend(np.split(y.numpy(), y.numpy().shape[0], axis=0))
  for x, y in trainloader:
    arr_x.extend(np.split(x.numpy(), x.numpy().shape[0], axis=0))
    arr_y.extend(np.split(y.numpy(), y.numpy().shape[0], axis=0))
  arr_x = np.array(arr_x)
  arr_y = np.array(arr_y)
  
  arr_x.flags.writeable = False
  arr_y.flags.writeable = False

  with open(os.path.join(args.ckpt_folder, 'hash.txt'), 'a') as f:
    f.write('train data x hash:' + hashlib.md5(arr_x.tobytes()).hexdigest() + '\n')
    f.write('train data y hash:' + hashlib.md5(arr_y.tobytes()).hexdigest() + '\n')

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_schedule(epoch), verbose=True)

def save_prediction(epoch, logs):
  pred_array = np.array([]).reshape(0, 10)
  for x, y in testloader:
    pred = model(x)
    pred_array = np.concatenate((pred_array, pred))

  pred_array = np.argmax(pred_array, axis=1)
  np.savetxt(os.path.join(args.ckpt_folder, f'pred{epoch}.txt'), pred_array)

def save_model(epoch, logs):
  if epoch in [9, 49, 99, 199]:
    model.save(os.path.join(args.ckpt_folder, f'ckpt{epoch}.h5'))

save_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_prediction, verbose=True)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(args.ckpt_folder, 'log.csv'))
ckpt_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)

keras.backend.clear_session()

model = Model()



optimizer = keras.optimizers.Adam(learning_rate=args.lr)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [csv_logger]
if args.save_ckpt:
  callbacks.append(ckpt_callback)
if args.save_pred:
  callbacks.append(save_callback)

get_weight_hash()
get_input_hash()

EPOCHS = 200

start = time.time()
_ = model.fit(trainloader,
          epochs=EPOCHS,
          validation_data=testloader,
          callbacks=callbacks)
end = time.time()
print("Network takes {:.3f} seconds to train".format(end - start))

# loss, accuracy = model.evaluate(testloader)
# print("Test Error Rate: ", round((1-accuracy)*100, 2), '%')
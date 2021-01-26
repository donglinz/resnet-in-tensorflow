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
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--tpu', action='store_true')
args = parser.parse_args()

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import resnet50
from tensorflow.keras.initializers import GlorotUniform

import os
if args.deterministic_tf:
  print('Enabling deterministic tensorflow operations and cuDNN...')
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

if args.tpu:
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))
  strategy = tf.distribute.TPUStrategy(resolver)

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
  
if args.tpu:  
  dataset = tfds.load('cifar10', data_dir='gs://donglin-datasets/')
else:
  dataset = tfds.load('cifar10')

trainloader, testloader = dataset['train'].cache(), dataset['test'].cache()

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = args.batch_size
IMG_SHAPE = 32



def preprocess_image(data):
  img = tf.cast(data['image'], tf.float32)
  img = img/255.

  return img, data['label']
if args.deterministic_input:
    trainloader = (
        trainloader
        .shuffle(1024, seed=0)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )
else:
    trainloader = (
        trainloader
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )

testloader = (
    testloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
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
  arr = []
  for w in model.weights:
    name = w.name
    w = w.numpy()
    w.flags.writeable = False
    arr.append(name + ' ' + hashlib.md5(w.tobytes()).hexdigest())
  tf.io.write_file(os.path.join(args.ckpt_folder, 'weighthash.txt'), '\n'.join(arr))
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
  tf.io.write_file(os.path.join(args.ckpt_folder, 'inputhash.txt'), 'train data x hash:' + hashlib.md5(arr_x.tobytes()).hexdigest() + '\n'+ 'train data y hash:' + hashlib.md5(arr_y.tobytes()).hexdigest() + '\n')


lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_schedule(epoch), verbose=True)

def save_prediction(epoch, logs):
  if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
    pred_array = np.array([]).reshape(0, 10)
    for x, y in testloader:
      pred = model(x)
      pred_array = np.concatenate((pred_array, pred))

    pred_array = np.argmax(pred_array, axis=1)
    tf.io.write_file(os.path.join(args.ckpt_folder, f'pred{epoch}.txt'), '\n'.join(map(lambda x: str(x), pred_array)))

def save_model(epoch, logs):
  if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
    tf.keras.models.save_model(model, os.path.join(args.ckpt_folder, f'ckpt{epoch}'), save_format='tf')

save_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_prediction, verbose=True)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(args.ckpt_folder, 'log.csv'))
ckpt_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)

keras.backend.clear_session()
if args.tpu:
  with strategy.scope():
    model = Model()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
  model = Model()
  optimizer = keras.optimizers.Adam(learning_rate=args.lr)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [csv_logger]
if args.save_ckpt:
  print('Adding model save ckpt callback...')
  callbacks.append(ckpt_callback)
if args.save_pred:
  print('Adding model save prediction callbaack...')
  callbacks.append(save_callback)

get_weight_hash()
get_input_hash()

start = time.time()
_ = model.fit(trainloader,
          epochs=args.epochs,
          validation_data=testloader,
          callbacks=callbacks)
end = time.time()
print("Network takes {:.3f} seconds to train".format(end - start))

# loss, accuracy = model.evaluate(testloader)
# print("Test Error Rate: ", round((1-accuracy)*100, 2), '%')

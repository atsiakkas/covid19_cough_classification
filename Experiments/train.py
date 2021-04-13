import os
import argparse
import ast
import datetime
from collections import Counter

from pathlib import Path
import librosa
import numpy as np
from absl import logging
import pandas as pd
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPool2D)
from tensorflow.keras.models import Model

from custom_layers.log_mel_spectrogram import LogMelSpectrogram
from custom_layers.spec_augment import SpecAugment

AUTOTUNE = tf.data.experimental.AUTOTUNE
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

########################################################################
#                        Arguments/parameters                          #
########################################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                 description='Trains an audio\
                                     classification model.')
parser.add_argument('--dataset', help='The dataset folder.')
parser.add_argument('--data_type', help='The type of data (tfrecords, wav, flac\
                    (default wav).')
parser.add_argument('--train_metadata',
                    help='Path to the training metadata CSV for calculating\
                        the class weights (and optionally creating a dataset).')
parser.add_argument('--valid_metadata',
                    help='Path to the validation metadata CSV creating\
                        a validation dataset).')
parser.add_argument('--test_metadata',
                    help='Path to the test metadata CSV creating\
                        a test dataset).')
parser.add_argument('--output_path',
                    help='Folder to output the trained model and logs.')
parser.add_argument('--model_name',
                    help='Model name. Used to select hyperparameters from CSV\
                        if such a CSV is provided (default audio_classifier')
parser.add_argument('--hyperparameters',
                    help='Optional path to a CSV containing the hyperparameters\
                        for training (default None')

# architectural arguments
group_arch = parser.add_argument_group('Architecture')
group_arch.add_argument('--dropout', help='(default .25)')
group_arch.add_argument('--weights',
                        help='Used in ResNet 50. Either None,\
                            path to weights or "image_net" (default None)')

# dataset arguments
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--n_classes',
                           help="Number of classes to classify (default 2)")
group_dataset.add_argument('--duration',
                           help="The duration of the audio files if fixed")
# training arguments
group_training = parser.add_argument_group('Training')

group_training.add_argument('--epochs',
                            help="Max trianing epochs (default 1000)")
group_training.add_argument('--batch_size', help="")
group_training.add_argument('--learning_rate', help="")
group_training.add_argument('--early_stop',
                            help="-1 is off, values >= 0 used\
                                as patience (default 20)")
group_training.add_argument('--resume',
                            help='Given the path to an existing model will\
                                resume the training of said model. Notice in\
                                such case architectural arguments are discarded.',
                            nargs='?',
                            default=None)
group_training.add_argument('--inference',
                            help='Whether to run the model in inference mode.',
                            default=False)
group_training.add_argument('--freeze_base',
                            help="Whether to freeze the base CNN architecture.\
                                (default False)")
group_training.add_argument('--load_weights',
                            help="Path to model from which to load weights.\
                                (default None)")

# data augmentation arguments
group_data_aug = parser.add_argument_group('Data augmentation')
group_data_aug.add_argument('--specaugment',
                            help="Whether to use SpecAugment (default True")
group_data_aug.add_argument('--time_warping',
                            help="The time warping parameter\
                                for Spec Augment (default 40")
group_data_aug.add_argument('--frequency_masking',
                            help="The frequency masking parameter\
                                for Spec Augment. (default 15")
group_data_aug.add_argument('--frequency_mask_num',
                            help="The number of frequency masks\
                                used by Spec Augment (default 2)")
group_data_aug.add_argument('--time_masking',
                            help="The time masking parameter\
                                for Spec Augment (default 70")
group_data_aug.add_argument('--time_mask_num',
                            help="The number of time masks\
                                used by Spec Augment (default 2")

# other arguments
group_misc = parser.add_argument_group('Other')
group_misc.add_argument('--verbose',
                        help="Verbosity level ranging from 0 (none)\
                            to 2 (max) (default 2)")

args = parser.parse_args()

params = {
    'temp': 186,  #input shape to ResNet50
    'model_name': 'audio_classifier',
    'hyperparameters': None, 
    'output_path': 'models_and_logs',
    'train_metadata': None,
    'valid_metadata': None,
    'test_metadata': None,
    'dataset': None,
    'data_type': 'wav',
    'n_classes': 2,
    'weights': None,
    'batch_size': 64,
    'epochs': 1000,
    'freeze_base': False,
    'early_stop': 20,
    'dropout': .025,
    'learning_rate': 0.001,
    # audio and spectrogram parameters:
    'sample_rate': 16000,
    'duration': 5,  # seconds
    'fft_size': 512,
    'hop_size': 256,
    'n_mels': 64,
    'f_min': 80,
    'a_min': 1e-6,
    # data augmentation parameters:
    'specaugment': True,
    'time_warping': 40,  # 80 or 40 in paper
    'frequency_masking': 15,  # 27 or 15 in paper
    'frequency_mask_num': 2,  # 1 or 2 in paper
    'time_masking': 70,  # 100 or 70 in paper
    'time_mask_num': 2,  # 1 or 2 in paper
    # other:
    'strides': 1
}

# Overwrite default parameters with user input parameters
string_params = [
    'dataset', 'data_type', 'output_path', 'model_name', 'train_metadata',
    'valid_metadata', 'test_metadata', 'weights','load_weights', 'hyperparameters'
]
skip_params = ['verbose', 'resume', 'inference']

# Process non-string args
for pair in args._get_kwargs():
    if pair[1] and pair[0] not in skip_params:
        params[pair[0]] = pair[1] if pair[0] in string_params \
                                  else ast.literal_eval(pair[1])

verbose = int(args.verbose or 2)

# If using CSV for hyperparameters instead of commandline arguments
if params['hyperparameters'] is not None:
    params_df = pd.read_csv(params['hyperparameters'])
    # Select the hyperparameters for the model we're interested in
    params_df = params_df[params_df.model_name == params['model_name']]
    if len(params_df) > 1:
        raise ValueError('Multiple rows in hyperparameter csv with the same name.')
    elif len(params_df) == 0:
        raise ValueError('Please supply a valid model name.')

    for key, value in params.items():
        if type(value) == int:
            params[key] = int(params_df[key].to_numpy())
        elif type(value) == float:
            params[key] = float(params_df[key].to_numpy())
        else:
            if str(params_df[key].to_numpy()[0]) == 'None':
                params[key] = None
            elif str(params_df[key].to_numpy()[0]).lower() == 'true':
                params[key] = True
            elif str(params_df[key].to_numpy()[0]).lower() == 'false':
                params[key] = False
            else:
                params[key] = str(params_df[key].to_numpy()[0])

########################################################################
#                      Tensorflow Dataset Operators                    #
########################################################################

def get_dataset(csv, split):
    if split == 'train':
        df = pd.read_csv(params['train_metadata'])
    elif split == 'validate':
        df = pd.read_csv(params['valid_metadata'])
    elif split == 'test':
        df = pd.read_csv(Path(params['test_metadata']))
    else:
        raise ValueError('Please specify train/validate/test for your dataset split')
    file_path_ds = tf.data.Dataset.from_tensor_slices(df.path)
    label_ds = tf.data.Dataset.from_tensor_slices(df.label)
    return tf.data.Dataset.zip((file_path_ds, label_ds))

def load_segment_audio(file_path):
    n_samples = params['sample_rate'] * params['duration']
    audio, _ = librosa.load(file_path.numpy(), sr=params['sample_rate'])   
    non_silent = librosa.effects.split(audio, top_db=25)
    audio_trimmed = np.empty(0)
    
    for i in non_silent:
        audio_trimmed = np.append(audio_trimmed, audio[i[0]: i[1]], axis=0)
    
    if len(audio_trimmed) > n_samples:
        audio_trimmed = audio_trimmed[:n_samples]

    elif len(audio_trimmed) < n_samples:
        audio_trimmed = np.pad(audio_trimmed, [0, (n_samples-len(audio_trimmed))])

    else: # audio_trimmed == n_samples:
        audio_trimmed = audio_trimmed

    return audio_trimmed

def load_segment_audio_wrapper(file_path, label):
    [audio,] = tf.py_function(load_segment_audio, [file_path], [tf.float64])
    return audio, label

def get_dataset_from_wavs(df, split, shuffle_buffer_size=1024, batch_size=params['batch_size']):
    ds = get_dataset(df, split)
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_segment_audio_wrapper, num_parallel_calls=AUTOTUNE)
    if split == 'train':
        # Repeat dataset forever
        ds = ds.repeat()
    # Prepare batches
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# TF Records
def _parse_batch(record_batch, sample_rate, duration):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'audio': tf.io.FixedLenFeature([n_samples], tf.float32),
        # 'audio': tf.io.FixedLenFeature([160000], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)

    return example['audio'], example['label']


def get_dataset_from_tfrecords(
        dataset_path,
        split='train',
        batch_size=params['batch_size'],  # was 64
        sample_rate=params['sample_rate'],
        duration=params['duration'],
        n_epochs=params['epochs']):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(dataset_path, '{}*.tfrecord'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB',
                                 num_parallel_reads=AUTOTUNE)
    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration), num_parallel_calls=AUTOTUNE)

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    if split == 'train':
        ds.shuffle(832) #TODO infer this number from the train CSV
        ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)


########################################################################
#                          Define Model                                #
########################################################################

def ResNet50(n_classes,
             sample_rate=params['sample_rate'],
             duration=params['duration'],
             fft_size=params['fft_size'],
             hop_size=params['hop_size'],
             n_mels=params['n_mels'],
             time_warping=params['time_warping'],
             frequency_masking=params['frequency_masking'],
             frequency_mask_num=params['frequency_mask_num'],
             time_masking=params['time_masking'],
             time_mask_num=params['time_mask_num']):
    n_samples = sample_rate * duration
    input_shape = (n_samples, )

    if params['weights'] == 'imagenet':
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=params['weights'],
            input_shape=(params['temp'], 64, 3)
        )  

    else:
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=params['weights'],
            input_shape=(params['temp'], 64, 1))

    x = Input(shape=input_shape, name='input', dtype='float32')
    y = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels)(x)
    if params['specaugment']:
        y = SpecAugment(time_warping, frequency_masking, frequency_mask_num,
                        time_masking, time_mask_num)(y)
    if params['weights'] == 'imagenet':
        y = tf.keras.layers.Concatenate()(
            [y, y, y])  # to make 3 channel from grayscale
        y = base_model(y)
    else:
        y = base_model(y)
    y = tf.keras.layers.GlobalAvgPool2D()(y)
    y = tf.keras.layers.Dropout(params['dropout'])(y) 
    if n_classes == 2:
        y = Dense(1, activation='sigmoid')(y)
    else:
        y = Dense(n_classes, activation='softmax')(y)

    return Model(inputs=x, outputs=y)


########################################################################
#                           Other Functions                            #
########################################################################


def log(message, message_verbose, sys_verbose):
    if message_verbose <= sys_verbose:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
              message)


def train_model(params, train_ds, valid_ds, model=None, output_path=''): 
    def get_class_weights(y):
        counter = Counter(y)
        majority = max(counter.values())
        return {
            cls: round(float(majority) / float(count), 2)
            for cls, count in counter.items()
        }

    df_train = pd.read_csv(params['train_metadata'])
    class_weights = get_class_weights(df_train['label'].values)

    df_valid = pd.read_csv(params['valid_metadata'])

    STEPS = tf.math.floor(len(df_train) / params['batch_size']) 
    VAL_STEPS = tf.math.floor(len(df_valid) / params['batch_size']) 
    print(f'STEPS: {STEPS}')
    print(f'VAL_STEPS: {VAL_STEPS}')

    model = ResNet50(params['n_classes'])

    if args.resume:
        if args.resume != '':
            model.load_weights(args.resume) #TODO this won't save the state of the optimizer
            print(f'Loaded weights from {args.resume} successfully. Training resuming.')

    if params['freeze_base'] == True:
        model.layers[4].trainable = False

    if params['n_classes'] == 2:
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['AUC'])
    else:
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'])
    model.summary()
    print(f'Class weights: {class_weights}')

    # prepare saving paths
    s_current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    log_name = s_current_time + '_' + params['model_name']
    logdir = os.path.join(output_path, 'logs', log_name)
    model_path = os.path.join(
        output_path, params['model_name'] + '.' + s_current_time + '.h5')

    # prepare callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir)  #, profile_batch=0)
    # tf 2.1 bug workaround
    tb_callback._log_write_dir = logdir

    if params['n_classes'] == 2:
        if args.resume:
            if args.resume != '':
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    model_path.replace('.h5', '.resume.h5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)

        else:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            patience=params['early_stop'],
            monitor='val_loss',
            verbose=1,
            mode='min',
            restore_best_weights=True)

    else:
        if args.resume:
            if args.resume != '':
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    model_path.replace('.h5', '.resume.h5'),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)

        else:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            patience=params['early_stop'],
            monitor='val_loss',
            verbose=1,
            mode='min',
            restore_best_weights=True)

    callbacks = [tb_callback, checkpoint_callback]
    if params['early_stop'] >= 0:
        callbacks.append(early_stop_callback)

    model.fit(train_ds,
              epochs=params['epochs'],
              steps_per_epoch=STEPS,
              validation_data=valid_ds, 
              validation_steps=VAL_STEPS,
              class_weight=class_weights,
              callbacks=callbacks)

    # prepare path outputs
    output_paths = [model_path]
    # add all log files
    for path, _, files in os.walk(logdir):
        for name in files:
            output_paths.append(os.path.join(path, name))

    return output_paths


########################################################################
#                               Script                                 #
########################################################################

if __name__ == '__main__':
    log('Loading dataset', 1, verbose)
    if params['data_type'] == 'wav':
        train_ds = get_dataset_from_wavs(params['train_metadata'], split='train') 
        valid_ds = get_dataset_from_wavs(params['valid_metadata'], split='validate')
        try:
            valid_ds = get_dataset_from_wavs(params['test_metadata'], split='test')
        except:
            print('No test set provided, continuing.')

    elif params['data_type'] == 'tfrecords':
        train_ds = get_dataset_from_tfrecords(params['dataset'], split='train')
        valid_ds = get_dataset_from_tfrecords(params['dataset'], split='validate')
        test_ds = get_dataset_from_tfrecords(params['dataset'], split='test')

    for key, value in params.items():
        print(key, value)

    log('Training model', 1, verbose)

    model = None

    output_files_paths = train_model(params,
                                     train_ds,
                                     valid_ds,
                                     model=model,
                                     output_path=params['output_path'])

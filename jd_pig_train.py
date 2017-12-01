import os
import numpy as np
import h5py
# Model
from keras.layers import Input, Flatten, Dropout, Dense, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D
from keras.layers import Multiply, Add, UpSampling2D, Lambda, GlobalAveragePooling2D
from keras import layers
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import rotate, resize
from skimage import io, filters
import math


def get_data_from_hdf5(file_local):
    f = h5py.File(file_local, "r")
    data_ = f['Data'][:]
    labels_ = f['labels'][:]
    # shuffle
    nb_data = len(data_)
    indices = np.arange(nb_data)
    np.random.shuffle(indices)
    data_ = data_[indices]
    labels_ = labels_[indices]
    f.close()
    return data_, labels_


def residual_unit(x, nb_filters, filters_diff=False):
    x = Activation('relu')(x)

    if filters_diff:
        residual = Conv2D(nb_filters, (1, 1), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
    else:
        residual = x

    x = SeparableConv2D(nb_filters, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(nb_filters, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = layers.add([x, residual])

    return x


def trunk(x, repeat, nb_filters, name):
    for i in range(repeat):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(nb_filters, (3, 3), padding='same', use_bias=False, name=name+str(i)+'_SepConv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(nb_filters, (3, 3), padding='same', use_bias=False, name=name+str(i)+'SepConv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(nb_filters, (3, 3), padding='same', use_bias=False, name=name+str(i)+'SepConv3')(x)
        x = BatchNormalization()(x)

        x = layers.add([x, residual])
    return x


def mask(x, nb_filters, name):
    # downsample 1
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=name+'downsample_1')(x)
    x = residual_unit(x, nb_filters)
    s1= residual_unit(x, nb_filters)

    # downsample 2
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=name+'downsample_2')(x)
    x = residual_unit(x, nb_filters)
    s2= residual_unit(x, nb_filters)

    # downsample 3
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=name+'downsample_3')(x)
    x = residual_unit(x, nb_filters)
    x = residual_unit(x, nb_filters)

    # upsample 3
    x = UpSampling2D(size=(2, 2))(x)
    x = layers.add([x, s2])
    x = residual_unit(x, nb_filters)

    # upsample 2
    x = UpSampling2D(size=(2, 2))(x)
    x = layers.add([x, s1])
    x = residual_unit(x, nb_filters)

    # upsample 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(nb_filters, (1, 1), padding='same', use_bias=False)(x)
    x = Conv2D(nb_filters, (1, 1), padding='same', use_bias=False)(x)

    # sigmoid
    x = Activation('sigmoid')(x)

    return x


# define train model
# Input (384, 384, 3)
# output (30, 1)
def model_class1(num_classes):
    input_tensor = Input(shape=(72, 72, 3))
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output_layer = Activation('softmax', name='predictions')(x)

    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, activation='elu')(x)

    # model
    model_ = Model(inputs=[input_tensor], outputs=output_layer)
    return model_


def crop_random(src_data, train_img_size):
    number = len(src_data)
    dst_data = np.empty([number, train_img_size, train_img_size, src_data.shape[-1]], dtype=np.float32)
    # Input is 72*128 56 gap 8 range(8, 48)
    for idx in np.arange(number):
        offset_col = np.random.randint(8, 48)
        dst_data[idx, :, :, :] = src_data[idx, :, offset_col:offset_col + train_img_size, :]
    return dst_data


def generate_train_mini_batches(src_data, src_labels, batch_size_t, train_img_size):
    """
    do data augment then get the train data[Num,480,480,channel]
    """
    nb_data = len(src_data)
    assert batch_size_t < nb_data

    # yield infinite src_data
    while True:
        # new id range(0,nb_data)
        indices = np.arange(nb_data)
        np.random.shuffle(indices)

        # get mini-batch images
        start_idx = 0
        while start_idx < nb_data:
            if start_idx+batch_size_t > nb_data:
                excerpt = indices[start_idx:]
            else:
                excerpt = indices[start_idx:start_idx + batch_size_t]
            start_idx += batch_size_t
            # remainder samples have smaller batch_size
            # batch_size_user = len(excerpt)

            # get mini-batch data
            data_batch = src_data[excerpt]
            labels_batch = src_labels[excerpt]
            # crop data to 72*72(crop center or crop random)
            data_train = crop_random(data_batch, train_img_size)

            yield([data_train], labels_batch)


if __name__ == "__main__":
    hdf5_local = 'jd_pig_train_data.h5'
    # Step1--read train data and labels
    data, labels = get_data_from_hdf5(hdf5_local)
    # 44250,72,128,3
    print(data.shape)

    model_name = 'XceptionModel'
    model_dir = os.path.join('CheckPoints', model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model = model_class1(num_classes=30)
    plot_model(model, to_file=os.path.join(model_dir, 'model.pdf'), show_shapes=True)

    batch_size = 25
    batch_valid_size = 50
    train_image_size = 72

    csv_fn = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')

    # Step2--train model
    # call backs
    check_pointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensor_board = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=batch_size,
                                         write_graph=False, write_grads=True, write_images=False)
    adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae', 'accuracy', 'categorical_crossentropy'])
    print('Set call backs success------------')

    train_history = model.fit_generator(generate_train_mini_batches(data[0:32000, :, :, :], labels[0:32000, :],
                                                                    batch_size, train_image_size),
                                        steps_per_epoch=math.ceil(32000*1.0 / batch_size),
                                        epochs=2048 * 2,
                                        validation_data=generate_train_mini_batches(data[32000:44250, :, :, :],
                                                                                    labels[32000:44250, :],
                                                                                    batch_valid_size,
                                                                                    train_image_size),
                                        validation_steps=math.ceil(12250*1.0 / batch_valid_size),
                                        callbacks=[check_pointer, csv_logger, tensor_board])

    pdb.set_trace()
    print(train_history)
    # Step3--test model and print result

# from keras.layers import BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D
# from keras import layers
# from keras.models import Model
import imageio
from skimage.transform import resize

# from keras.utils import plot_model
# from keras import backend as K
# from keras import optimizers
# from keras import callbacks
import os
import numpy as np
import sys
import h5py


# define train model
# Input (384, 384, 3)
# output (30, 1)
# def model_class1():
#     input_tensor = Input(shape=(384, 384, 3))
#
#     x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input_tensor)
#
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(64, (3, 3), use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)
#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])
#
#     residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])
#
#     residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])
#
#     # attention block 1
#     t = trunk(x, repeat=2, nb_filters=728, name=name + '_trunk_1_')
#     m = mask(x, nb_filters=728, name=name + '_mask_1_')
#     x = Multiply()([t, m])
#     x = Add()([t, x])
#
#     # attention block 2
#     t = trunk(x, repeat=3, nb_filters=728, name=name + '_trunk_2_')
#     m = mask(x, nb_filters=728, nam e=name + '_mask_2_')
#     x = Multiply()([t, m])
#     x = Add()([t, x])
#
#     # attention block 3
#     t = trunk(x, repeat=3, nb_filters=728, name=name + '_trunk_3_')
#     m = mask(x, nb_filters=728, name=name + '_mask_3_')
#     x = Multiply()([t, m])
#     x = Add()([t, x])
#
#     # end
#     residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)
#
#     x = Activation('relu')(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])
#
#     x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     # post
#     x = Conv2D(512, (7, 7), use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(128, (5, 5), use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     # predict
#     x = Flatten()(x)
#     x = Dense(30, activation='linear')(x)
#
#     # model
#     model = Model(inputs=[input_tensor], outputs=x)
#     return model


def get_train_data(file_local):
    data_ = []
    labels_ = []
    for idx in range(0, 30):
        file_temp = os.path.join(file_local, str(idx + 1) + '.mp4')
        # frame size is (720, 1280, 3)
        label_temp = np.zeros(30)
        label_temp[idx] = 1  # set the image id
        video_temp = imageio.get_reader(file_temp, 'ffmpeg')
        for video_frame in range(0, video_temp.get_length() - 1):
            if (video_frame % 2) == 0:
                sys.stdout.write('\r>> Read image from video frame ID %d/%d' % (idx, video_frame))
                sys.stdout.flush()
                image_temp = video_temp.get_data(video_frame)
                # resize image to reduce calculate amount
                image_temp = resize(image_temp, (72, 128, 3))
                data_.append(image_temp)
                labels_.append(label_temp)

    data_ = np.asarray(data_)
    labels_ = np.asarray(labels_)

    print("\n creating hdf5 file...")
    print("\n data is (IDX,72,128,3) labels is (IDX,30)...")
    f = h5py.File('jd_pig_train_data.h5', "w")
    dst_data = f.create_dataset('Data', data_.shape, np.float32)
    dst_data[:] = data_[:]
    dst_labels = f.create_dataset('labels', labels_.shape, np.float32)
    dst_labels[:] = labels_[:]
    f.close()

    return data_, labels_

if __name__ == "__main__":
    # include 30 pig mp4 file named as ID.mp4
    train_local = 'Pig_Identification_Qualification_Train/train'
    # include 3000 JPG pigs in difference size
    test_local = 'Pig_Identification_Qualification_Test_A/test_A'

    # Step1--read train data and labels
    data, labels = get_train_data(train_local)
    print(data.shape)

    # batch_size = 5
    # batch_valid_size =10
    # train_image_size = 360
    # model_name = 'XceptionModel'
    # model_dir = os.path.join('CheckPoints', model_name)
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    #
    # csv_fn = os.path.join(model_dir, 'train_log.csv')
    # checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')
    # # Step1--read train data and deal the data to the same size
    # # input is(384,384,3) label is (1*30)1-rows,30-cols
    # # All the data is(N,384,384,3)(N,30)
    #
    # # Step2--train model
    # # call backs
    # check_pointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    # csv_logger = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    # tensor_board = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=10,
    #                                      write_graph=False, write_grads=True, write_images=False)
    #
    # model.compile(optimizer=adam, loss=eval_failed_mae_loss,
    #               metrics=['mae', loss_2cm, eval_std, eval_max, eval_real_max, eval_argmax, eval_failed_rate])
    # print('Set call backs success------------')
    # Step3--test model and print result

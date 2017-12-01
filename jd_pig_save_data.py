import imageio
from skimage.transform import resize

import os
import numpy as np
import sys
import h5py


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



import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
from jd_pig_train import model_class1
import pandas as pd
import sys


def crop_center(src_data, train_img_size):
    rows, cols, _ = src_data.shape
    # dst_data = np.empty([train_img_size, train_img_size, src_data.shape[-1]], dtype=np.float32)
    if rows < cols:
        dst_cols = int(cols * 1.0 / rows * train_img_size)
        src_data = resize(src_data, (train_img_size, dst_cols, 3))
        gap = int((dst_cols - train_img_size) / 2.0)
        dst_data = src_data[:, gap:gap + train_img_size, :]
    else:
        dst_rows = int(rows * 1.0 / cols * train_img_size)
        src_data = resize(src_data, (dst_rows, train_img_size, 3))
        gap = int((dst_rows - train_img_size) / 2.0)
        dst_data = src_data[gap:gap + train_img_size, :, :]
    return dst_data


if __name__ == "__main__":
    file_path = 'Pig_Identification_Qualification_Test_A/test_A'
    file_list = os.listdir(file_path)
    ID_List = []
    ImageTest = []
    for idx in np.arange(0, len(file_list)):
        sys.stdout.write('\r>> Read image from local %d' % idx)
        sys.stdout.flush()
        file_local_t = os.path.join(file_path, file_list[idx])
        image_temp = io.imread(file_local_t)
        dst_temp = crop_center(image_temp, train_img_size=72)
        id_t, _, _ = file_list[idx].partition('.')
        ID_List.append(int(id_t))
        ImageTest.append(dst_temp)
    ID_List = np.asarray(ID_List)
    ImageTest = np.asarray(ImageTest)

    model = model_class1(30)
    model.load_weights('checkpoint.16-0.00.hdf5')
    result = model.predict(ImageTest)

    Result_labels = []
    for idx in np.arange(0, len(ID_List)):
        for row in np.arange(0, 30):
            Result_labels.append([ID_List[idx], row + 1, result[idx, row]])
            # if result[idx, row] == result[idx, :].max():
            #     Result_labels.append([ID_List[idx], row + 1, 1])
            # else:
            #     Result_labels.append([ID_List[idx], row + 1, 0])

    Result_labels = np.asarray(Result_labels)

    df = pd.DataFrame(Result_labels)
    df.to_csv("my_class1_result4.csv", index=False)
    # print(result)

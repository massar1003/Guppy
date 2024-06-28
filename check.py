import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# CNNのモデルを定義する
def def_model(in_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(32,
              kernel_size=(3, 3),
              activation='relu',
              input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

# コンパイル済みのCNNのモデルを返す
def get_model(in_shape, nb_classes):
    model = def_model(in_shape, nb_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])
    return model

im_rows = 64 # 画像の縦ピクセルサイズ
im_cols = 64 # 画像の横ピクセルサイズ
im_color = 3 # 画像の色空間
in_shape = (im_rows, im_cols, im_color)
nb_classes = 3

LABELS = ["オス","メス"]

# 保存したCNNモデルを読み込む
model = get_model(in_shape, nb_classes)
model.load_weights('image/photos-model-light.hdf5')

def check_photo(path):
    # 画像を読み込む
    img = Image.open(path)
    img = img.convert("RGB") # 色空間をRGBに
    img = img.resize((im_cols, im_rows)) # サイズ変更
    plt.imshow(img)
    plt.show()
    # データに変換
    x = np.asarray(img)
    x = x.reshape(-1, im_rows, im_cols, im_color)
    x = x / 255

    # 予測
    pre = model.predict([x])[0]
    idx = pre.argmax()
    per = int(pre[idx] * 100)
    return (idx, per)

def check_photo_str(path):
    idx, per = check_photo(path)
    # 答えを表示
    print("この写真は、", LABELS[idx], "だと思います。")
    print("可能性は ", per, "% 。")

if __name__ == '__main__':
    check_photo_str('グッピーオス/IMG_20200519_213148.jpg')
    print("※ オスです。")
    print()
    check_photo_str('グッピーメス/16433526737_0df4bc840a_w.jpg')
    print("※ メスです。")
    print()
    check_photo_str('グッピーメス/4686255259_99e3d185fd_n.jpg')
    print("※ メスです。")
    print()
    check_photo_str('グッピーオス/P6250016.jpg')
    print("※ オスです。")
    print()
    check_photo_str('グッピーオス/images (2).jpg')
    print("※ オスです。")
    print()
    check_photo_str('グッピーメス/IMG_20200519_213128.jpg')
    print("※ メスです。")
    print()
    check_photo_str('グッピーメス/9222928992_c734b127d2_w.jpg')
    print("※ メスです。")
    print()
    check_photo_str('グッピーメス/7852134236_7b6c79ac26_n.jpg')
    print("※ メスです。")
    print()
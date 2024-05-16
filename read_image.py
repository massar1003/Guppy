import numpy as np
from PIL import Image
import os, glob, random

outfile = "/content/drive/MyDrive/人工知能/人工知能1/自分用/image/photos.npz" # 保存ファイル名
max_photo = 200 # 利用する写真の枚数
photo_size = 64 # 画像サイズ
x = [] # 画像データ
y = [] # ラベルデータ

def main():
    # 保存先フォルダのパス
    output_folder = os.path.dirname(outfile)

    # フォルダが存在しない場合は作成する
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 各画像のフォルダを読む
    glob_files("/content/drive/MyDrive/人工知能/人工知能1/自分用/グッピー/グッピーオス", 0)
    glob_files("/content/drive/MyDrive/人工知能/人工知能1/自分用/グッピー/グッピーメス", 1)

    # ファイルへ保存
    np.savez(outfile, x=x, y=y)
    print("保存しました:" + outfile, len(x))

# path以下の画像を読み込む
def glob_files(path, label):
    files = glob.glob(path + "/*.jpg")
    random.shuffle(files)

    # 各ファイルを処理
    num = 0
    for f in files:
        if num >= max_photo: break
        num += 1

        # 画像ファイルを読む
        img = Image.open(f)
        img = img.convert("RGB") # 色空間をRGBに
        img = img.resize((photo_size, photo_size)) # サイズ変更
        img = np.asarray(img)
        x.append(img)
        y.append(label)

main()
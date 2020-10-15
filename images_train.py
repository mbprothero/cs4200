# check Pillow version number
# import PIL
# print('Pillow Version:', PIL.__version__)

# from PIL import Image
# from numpy import asarray
# Load Image
# image = Image.open('Michael_Prothero.jpg')
# print(image.format)
# print(image.mode)
# print(image.size)

# show the image
# image.show()






import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR,"images")

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            print(path)

import numpy as np
import os
import glob
import tensorflow as tf
import PIL.Image

ROOT_DIR = 'data'
DATA_LIST = glob.glob(ROOT_DIR + '\\*\\*.jpg')
IMG_SIZE = 256
RESIZED_DIR = "resized"

def get_label(path):
    return path.split('\\')[1]

def get_file_name(path):
    return path.split('\\')[2]

def make_dir(label, keyword="aug"):
    dir = ROOT_DIR + "\\" +label + "\\"+ keyword + "\\"
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("--> ", dir)
    return dir

def resize(org_img):
    image = PIL.Image.open(org_img)
    resized = image.resize((IMG_SIZE,IMG_SIZE))
    dir = make_dir(get_label(org_img), RESIZED_DIR)
    resized.save(dir + get_file_name(org_img), "JPEG", quality=100)
    return resized

def write_tf(tf_img, org_img, keyword):
    dir = make_dir(get_label(org_img), keyword)
    enc = tf.io.encode_jpeg(tf_img, quality = 100)
    tf.io.write_file(dir+"\\"+dir.split('\\')[2][0:2]+"-"+get_file_name(org_img), enc)

def augment(res_img):
    image_string = tf.io.read_file(res_img)
    image = tf.image.decode_jpeg(image_string, channels = 3)
    
    flipped = tf.image.flip_left_right(image)
    write_tf(flipped, res_img, "flipped")

    rotated = tf.image.rot90(image)
    write_tf(rotated, res_img, "rotated")

    grayscaled = tf.image.rgb_to_grayscale(image)
    write_tf(grayscaled, res_img, "grayscaled")

    saturated = tf.image.adjust_saturation(image, 3)
    write_tf(saturated, res_img, "saturated")

    bright = tf.image.adjust_brightness(image, 0.4)
    write_tf(bright, res_img, "bright")


for org_img in DATA_LIST:
   resize(org_img)
   res_img = make_dir(get_label(org_img), RESIZED_DIR) + get_file_name(org_img)
   augment(res_img)

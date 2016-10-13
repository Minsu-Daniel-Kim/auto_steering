__author__ = 'DanielMinsuKim'


import scipy.misc
import os
from PIL import Image
import cv2
import pickle
import json


n_bundle = 2

LOGDIR = 'driving_dataset2'

def read_data_from_processed_pickle(pickle_data):
    print("read processed pickle...")
    with open("../processed_pickle/%s" % pickle_data, 'rb') as handle:
        # data = pickle.load(handle,encoding='latin1')
        data = pickle.load(handle)
        return data

# dataset = read_data_from_processed_pickle("")
path = os.getcwd() + "/../processed_pickle"
print("path: %s" % path)
processed_pickles = [item for item in os.listdir(path) if item.endswith(".pickle")]
processed_pickles = processed_pickles[:n_bundle]

bundle = []
for item in processed_pickles:

    bundle.append(read_data_from_processed_pickle(item))
# if os.path.exists(LOGDIR):
#     os.removedirs(LOGDIR)
os.makedirs(LOGDIR)


bundle_image = []
bundle_label = []

for image, label in bundle:
    bundle_image.extend(image)
    bundle_label.extend(label)


i = 0
data_label = {}
le = len(bundle_image)


while le > 0:

    file_name = "%s.jpg" % bundle_image[i]['key']
    checkpoint_path = os.path.join(LOGDIR, file_name)

    if i % 1000 == 0:

        print(checkpoint_path)

    im = Image.fromarray(bundle_image[i]['image'])
    im.save(checkpoint_path)


    label_name = "%s.jpg" % bundle_label[i]['key']
    data_label[label_name] = bundle_label[i]['label'][0]

    i += 1
    le -= 1

with open(os.path.join(LOGDIR, 'data.json'), 'w') as outfile:
    json.dump(data_label, outfile)
import tensorflow as tf
import scipy.misc
import model
import cv2
import os
import pickle
import time
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('tesla_wheel.png',0)
img = scipy.misc.imresize(img, [300, 300])
rows,cols = img.shape

smoothed_angle = 0

i = 0


def read_data_from_processed_pickle(pickle_data):
    print("read processed pickle...")
    with open("../db/processed_pickle/%s" % pickle_data, 'rb') as handle:
        data = pickle.load(handle,encoding='latin1')
        # data = pickle.load(handle)
        return data

path = os.getcwd() + "/../db/driving_dataset2"
print("path: %s" % path)
processed_pickles = [item for item in os.listdir(path) if item.endswith(".jpg")]

while(cv2.waitKey(10) != ord('q')):
    time.sleep(0.01)
    full_image = scipy.misc.imread("../db/driving_dataset2/" + processed_pickles[i], mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    # degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] / scipy.pi
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    # degrees = degrees
    call("clear")
    print("Estimated Steering Angle: " + str(degrees) + " degrees")
    cv2.imshow("Video View", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()

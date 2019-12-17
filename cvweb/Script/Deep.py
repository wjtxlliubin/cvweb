import os  # file path
from io import BytesIO  # read image from url
import tarfile  # process weight file
import tempfile
from six.moves import urllib  # network lib
import cv2
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image  # Image可以进行二进制文件操作，没有用到opencv，因为opencv是一个非常重量级的库，contrib

import tensorflow as tf

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef().FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]



def run_visualization(original_im):
    MODEL = DeepLabModel(
        '/Users/admin/PycharmProjects/Cvdeeplabv3p/cvweb/Script/deeplabv3_pascal_train_aug_2018_01_04.tar.gz')

    resized_im, seg_map = MODEL.run(original_im)  # Inference，U-Net simple， U-Net base，deeplab v3+
    output = label_to_color_image(seg_map).astype(np.uint8)
    input = np.asarray(resized_im)
    src_h, src_w, channel = output.shape
    dst_img = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    for i in range(channel):
        for dst_y in range(src_h):
            for dst_x in range(src_w):
                if output[dst_y, dst_x, i] != 0:
                    dst_img[dst_y, dst_x, i] = input[dst_y, dst_x, i]
                else:
                    if i == 0:
                        dst_img[dst_y, dst_x, i] = 255
                    else:
                        dst_img[dst_y, dst_x, i] = 0
    # return np.asarray(resized_im)
    # return label_to_color_image(seg_map).astype(np.uint8)
    return Image.fromarray(dst_img)

if __name__ == '__main__':
    original_im = Image.open('/Users/admin/PycharmProjects/cv-learn/CvLesson6/timg.jpg')
    run_visualization(original_im)
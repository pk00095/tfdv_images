import tensorflow as tf
import glob, os, tqdm
from PIL import Image
import base64

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image_string, label, height, width, channels, image_format):
  #image_shape = tf.image.decode_jpeg(image_string).shape
  #mask_shape = tf.image.decode_png(mask_string).shape

  feature = {
      'image/raw': _bytes_feature(image_string),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(channels),
      'image/label': _bytes_feature(label.encode('utf-8')),
      'image/format': _bytes_feature(image_format.encode('utf-8'))
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))



def create_tfrecords():

  base_dir = '/home/segmind/Desktop/test/tfdv/bikes_persons_dataset'
  outpath = '/home/segmind/Desktop/test/tfdv/bikes_persons_dataset/dataset.tfrecord'

  with tf.io.TFRecordWriter(outpath) as writer :

      for folder in glob.glob(os.path.join(base_dir,'*')):

        if os.path.isdir(folder):

            LABEL = os.path.basename(folder)
            print('Processing images for label :: {}'.format(LABEL))

            for image_file in tqdm.tqdm(glob.glob(os.path.join(folder,'*.*'))):

              #assert image_file.endswith('.jpg'),'required `.jpg` image got instead {}'.format(image_file)
              image_string = open(image_file, 'rb').read()

              img = Image.open(image_file)
              image_format = img.format
              width, height = img.size

              num_channels = len(img.getbands())

              tf_example = image_example(
                image_string=image_string.encode('utf-8'),
                label=LABEL,
                height=height, 
                width=width, 
                channels=num_channels,
                image_format=image_format)

              writer.write(tf_example.SerializeToString())
            #pbar.update(1)


if __name__ == '__main__':
  create_tfrecords()
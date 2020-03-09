import tensorflow as tf

from PIL import Image
import xml.etree.ElementTree as ET
import glob, os, tqdm


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_list_feature(value):
  """Returns a bytes_list from a list of string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
  """Returns a float_list from a list of float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
  """Returns an int64_list from a list of bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tf_example(data):

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(data['height']),
      'image/width': _int64_feature(data['width']),
      'image/channels': _int64_feature(data['channels']),
      'image/encoded': _bytes_feature(data['image']),
      'image/format': _bytes_feature(data['format']),
      'image/object/bbox/xmin': _float_list_feature(data['xmin']),
      'image/object/bbox/xmax': _float_list_feature(data['xmax']),
      'image/object/bbox/ymin': _float_list_feature(data['ymin']),
      'image/object/bbox/ymax': _float_list_feature(data['ymax']),
      'image/f_id': _int64_feature(data['f_id']),
      'image/object/class/label': _bytes_list_feature(data['labels']),
  }))

  return tf_example


def statistics_example(data):

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/object/bbox/height': _float_feature(data['height']),
      'image/object/bbox/width': _float_feature(data['width']),
      'image/object/class/label': _bytes_feature(data['label']),
  }))

  return tf_example


def get_image_and_annotations(image_file, file_id, label_file_pointer, outpath, xml_file=None):
    
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    labels = []

    img = Image.open(image_file)
    image_format = img.format
    width, height = img.size

    num_channels = len(img.getbands())

    image_string = open(image_file, 'rb').read()

    if xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            label = obj.find('name').text

            if label not in label_file_pointer:
              label_file_pointer[label] = tf.io.TFRecordWriter(os.path.join(outpath,label)+'.tfrecord')

            label_encoded = label.encode('utf-8')
            labels.append(label_encoded)

            bbox = obj.find('bndbox')

            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            xmin_list.append(x1)
            ymin_list.append(y1)
            xmax_list.append(x2)
            ymax_list.append(y2)

            stats_feature = {
              'height':y2-y1,
              'width':x2-x1,
              'label':label_encoded
            }

            stats_example = statistics_example(data=stats_feature)
            label_file_pointer[label].write(stats_example.SerializeToString())

    feature =  {
            'height' : height,
            'width' : width,
            'xmin' : xmin_list,
            'ymin' : ymin_list,
            'xmax' : xmax_list,
            'ymax' : ymax_list,
            'f_id' : file_id,
            'labels' : labels,
            'image' : image_string,
            'channels': num_channels,
            'format':image_format.encode('utf-8')
            }

    return create_tf_example(feature), label_file_pointer


def create_tfrecord(image_dir, xml_dir, outpath, outname='train.tfrecord', debug=False):

    label_file_pointer = dict()

    with tf.io.TFRecordWriter(os.path.join(outpath,outname)) as writer :

        for index,image_file in enumerate(tqdm.tqdm(glob.glob(os.path.join(image_dir,'*.*'))), 1):
            xml_file = image_file.replace(image_dir, xml_dir).replace('.jpg','.xml')
            if not os.path.isfile(xml_file):
              xml_file=None
            tf_example, label_file_pointer = get_image_and_annotations(image_file, 
              file_id=index, 
              label_file_pointer=label_file_pointer, 
              outpath=outpath, 
              xml_file=xml_file)
            writer.write(tf_example.SerializeToString())

        for label_stats_pointer_file in label_file_pointer.values():
          label_stats_pointer_file.close()



def test():
  image_dir = '/home/segmind/Desktop/test/tfdv/HardHat/Hardhat/Train/JPEGImage'
  xml_dir = '/home/segmind/Desktop/test/tfdv/HardHat/Hardhat/Train/Annotation'
  outpath = '/home/segmind/Desktop/test/tfdv/HardHat'
  create_tfrecord(image_dir, xml_dir, outpath, outname='hardhat.tfrecord', debug=True)

if __name__ == '__main__':
  test()
import os
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags

from siamese.io.path_definition import get_project_dir, get_tf_record_default_parameters


default_parameters = get_tf_record_default_parameters()

FLAGS = flags.FLAGS

flags.DEFINE_string('train_csv_path', default_parameters['TRAIN_CSV_PATH'], 'Training data csv file path')
flags.DEFINE_string('val_csv_path', default_parameters['VAL_CSV_PATH'], 'Validation data csv file path')
flags.DEFINE_string('test_csv_path', default_parameters['TEST_CSV_PATH'], 'Testing data csv file path')
flags.DEFINE_string('image_directory', default_parameters['IMAGE_DIRECTORY'], 'directory in which images are stored')
flags.DEFINE_integer('num_shards', default_parameters['NUM_SHARDS'], 'Number of shards in output data')


def _get_image_files_and_conditions(csv_path: str, image_dir: str) -> Tuple[List, List, List]:

    """
    Args:
        csv_path: path to the image metadata csv file
        image_dir: directory storing the images

    Returns:

    """

    with tf.io.gfile.GFile(csv_path, 'rb') as csv_file:
        df = pd.read_csv(csv_file)
        df.set_index('hash', inplace=True)

        image_paths = [f"{image_dir}/{cdn_url}" for cdn_url in df['cdn_url_legacy'].values]
        file_ids = [idx for idx in df.index]
        conditions = [df.loc[index].to_dict() for index in df.index]

    return image_paths, file_ids, conditions


def _write_tfrecord(output_prefix: str, image_paths: List, file_ids: List, conditions: List):

    """
    Read image files and write image and metadata into TFRecord files

    Args:
        output_prefix:
        image_paths: paths to images which will be converted
        file_ids: image unique ids
        conditions: list of di

    Returns:

    """

    if not len(image_paths) == len(file_ids) == len(conditions):

        raise ValueError(f'length if image_paths, file_ids, and conditions should be tha same,'
                         f'but they are {len(image_paths)}, {len(file_ids)}, {len(conditions)}, respectively')

    spacing = np.linspace(0, len(image_paths), FLAGS.num_shards + 1, dtype=np.int)

    dir_train = f"{get_project_dir()}/data/train"

    if not os.path.isdir(dir_train):
        os.makedirs(dir_train)

    for shard in range(FLAGS.num_shards):
        output_file = f"{dir_train}/{output_prefix}-{shard:05d}-of-{FLAGS.num_shards:05d}"
        writer = tf.io.TFRecordWriter(output_file)

        print(f"Processing shard {shard} and writing file {output_file}")

        for i in range(spacing[shard], spacing[shard+1]):
            image_buffer, height, width = _process_image(image_paths[i])
            example = _convert_to_example(file_ids[i], image_buffer, height, width, conditions[i])
            writer.write(example.SerializeToString())
        writer.close()


def _int64_feature(value: Union[int, List]):

    """
    Returns an int64 list from a bool/enum/int/uint

    Args:
        value:

    Returns:
    """

    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):

    """
    Returns a bytes list from a string byte

    Args:
        value:

    Returns:
    """

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(file_id: str, image_buffer, height: int, width: int, condition: Optional[Dict]=None):

    """
    Build an example proto for the given inputs.
    
    Args:
        file_id: unique id of the given image file
        image_buffer: 
        height: image height in pixels
        width: image width in pixels
        condition: Optional; image metadata 
    
    Return: 
    """

    if condition:
        c = [(k, v) for k, v in condition.items()]
        c.sort()
        c = [v.encode('utf-8') for _, v in c]
    else:
        c = []

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    features = {'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
                'image/channels': _int64_feature(channels),
                'image/format': _bytes_feature(image_format.encode('utf-8')),
                'image/id': _bytes_feature(file_id.encode('utf-8')),
                'image/encoded': _bytes_feature(image_buffer),
                'image/class/conditions': _bytes_feature(c)}

    return tf.train.Example(features=tf.train.Features(features=features))


def _process_image(filename: str) -> Tuple:

    """
    Process a sinle image file

    Args:
        filename: path to an image file

    Returns:

    """

    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Check that image has RGB channels

    # Decode the RGB jpeg
    image = tf.io.decode_jpeg(image_data, channels=3)

    if len(image.shape) != 3:
        raise ValueError(f'The parsed image number of dimension is not 3 but {image.shape}')

    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f'The parsed image channel is not 3 but {channels}')

    return image_data, height, width


def _build_tfrecord_dataset(name: str, image_dir: str):

    """
    Build a TFRecord dataset

    Args:
        name:
        image_dir: directory storing the images
    """

    assert name in ['train', 'val', 'test']

    if name == 'train':
        path_to_csv = FLAGS.TRAIN_CSV_PATH
    elif name == 'val':
        path_to_csv = FLAGS.VAL_CSV_PATH
    else:
        path_to_csv = FLAGS.TEST_CSV_PATH

    image_paths, file_ids, conditions = _get_image_files_and_conditions(path_to_csv, image_dir)

    _write_tfrecord(name, image_paths, file_ids, conditions)


def main():

    _build_tfrecord_dataset('train', FLAGS.image_directory)
    _build_tfrecord_dataset('val', FLAGS.image_directory)
    _build_tfrecord_dataset('test', FLAGS.image_directory)


if __name__ == "__main__":

    app.run(main)

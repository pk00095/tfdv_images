import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_data_validation as tfdv

tfrecord_path = '/home/segmind/Desktop/test/tfdv/bikes_persons_dataset/dataset.tfrecord'

semantic_stats_options = tfdv.StatsOptions(enable_semantic_domain_stats=True)

#exit()

stats = tfdv.generate_statistics_from_tfrecord(
	data_location=tfrecord_path, 
	stats_options=semantic_stats_options)

#print(stats)
tfdv.visualize_statistics(stats)
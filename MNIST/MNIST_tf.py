import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

log_dir = 'C:/Users/nktru/OneDrive/Desktop/MNIST'
feature_log = 'C:/Users/nktru/OneDrive/desktop/MNIST/mnist_features.tsv'
label_log = 'C:/Users/nktru/OneDrive/desktop/MNIST/mnist_labels.tsv'
image_log = 'C:/Users/nktru/OneDrive/Desktop/MNIST/sprite.png'
X_mnist = pd.read_csv(feature_log, sep='\t').values
y_mnist = pd.read_csv(label_log, sep='\t').values

features = tf.Variable(X_mnist, name='features')
# Creating a checkpoint from embedding, filename and key are name of the tensor
checkpoint = tf.train.Checkpoint(embedding=features)
checkpoint.save('C:/Users/nktru/OneDrive/Desktop/MNIST/embedding.ckpt')

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding.metadata_path = label_log
embedding.sprite.image_path = image_log
embedding.sprite.single_image_dim.extend([28,28])

projector.visualize_embeddings(log_dir, config)

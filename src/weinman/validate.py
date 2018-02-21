# CNN-LSTM-CTC-OCR
# Copyright (C) 2017, 2018 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn
print tf.__version__
import mjsynth
import model
from time import time
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','/home/loitg/debugtf/model_version4_total/',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('device','/device:GPU:0',
                           """Device for graph placement""")
tf.app.flags.DEFINE_string('imgsdir','/tmp/ap_samples/',
                           """Image to predict""")
tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers

def _get_image(filename):
    """Load image data for placement in graph"""
    image = Image.open(filename) 
    image = np.array(image)
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    if len(image.shape) < 3:
        image = image[:,:,np.newaxis] # so just extract first channel, preserving 3D shape
    else:
        image = image[:,:,:1]
    
    return image


def _preprocess_image(image):

    # Copied from mjsynth.py. Should be abstracted to a more general module.
    
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    #first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    #image = tf.concat([first_row, image], 0)

    return image


def _get_input(bucket_size):
    """Set up and return image and width placeholder tensors"""

    # Raw image as placeholder to be fed one-by-one by dictionary
    image = tf.placeholder(tf.uint8, shape=[bucket_size,32, None, 1])
    width = tf.placeholder(tf.int32,shape=[bucket_size,]) # for ctc_loss

    return image,width


def _get_output(rnn_logits,sequence_length):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
        predictions,probs = tf.nn.ctc_greedy_decoder(rnn_logits, 
                                                   sequence_length,
                                                   #beam_width=128,
                                                   #top_paths=3,
                                                   merge_repeated=True)
    dts = [tf.sparse_tensor_to_dense(dt, default_value=-1) for dt in predictions]
    return dts


def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)
    config.gpu_options.allow_growth = True
    return config


def _get_checkpoint(modeldir=FLAGS.model):
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(modeldir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path


def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([mjsynth.out_charset[c] for c in labels])
    return string

def main(argv=None):
    bs=3
    with tf.Graph().as_default():
        with tf.device('/device:CPU:0'):
            image,width = _get_input(bs) # Placeholder tensors
 
            proc_image = _preprocess_image(image)

        with tf.device('/device:CPU:0'):
            features,sequence_length = model.convnet_layers( proc_image, width, 
                                                             mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
        with tf.device('/device:CPU:0'):
            predictions = _get_output( logits,sequence_length)

            session_config = _get_session_config()
            restore_model = _get_init_trained()
        
            init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            restore_model(sess, _get_checkpoint()) # Get latest checkpoint
            print image, width
            print predictions            
            tt = time()
            sortedlist = sorted(os.listdir(FLAGS.imgsdir), key=lambda x:(len(x),x))
            for filename in sortedlist:
                if filename[-3:].upper() != 'JPG': continue
                line = os.path.join(FLAGS.imgsdir, filename)
                print line
                # Eliminate any trailing newline from filename
                image_data = _get_image(line.rstrip())
                image_data = np.array([image_data]*bs)
                w = image_data.shape[2]
                ws = np.array([w]*bs)
                
                
                # Get prediction for single image (isa SparseTensorValue)
                p = sess.run(predictions,{ image: image_data, 
                                                 width: ws} )
                print p[0].shape
                print p[0]
#                 print(str(time()-tt) + ':' + _get_string(output))

if __name__ == '__main__':
    tf.app.run()

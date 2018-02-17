# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading
from time import time, sleep

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from PIL import Image
import numpy as np


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '10.88.96.154:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 `~!@#$%^&*()-=_+[]{};'\\:\"|,./<>?"
def _get_image(filename):
    """Load image data for placement in graph"""
    image = Image.open(filename) 
    image = np.array(image)
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    image = image[:,:,np.newaxis] # so just extract first channel, preserving 3D shape

    return image

class _ResultCounter(object):
    """Counter for the prediction results."""
    
    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
    
    def inc_error(self):
        with self._condition:
            self._error += 1
    
    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()
    
    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()
    
    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
        return self._error / float(self._num_tests)
    
    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

class ResultAccumulator(object):
    def __init__(self, hostport):
        self.host, self.port = hostport.split(':')
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
    
    def predict_batch(self, batch):
        batchsize = batch.shape[0]
        lens = [batch.shape[2]]*batchsize
        lens = np.array(lens, dtype=np.int32)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'clreceipt'
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(batch, shape=batch.shape))
        request.inputs['width'].CopyFrom(
        tf.contrib.util.make_tensor_proto(lens, shape=[batchsize]))
        try:
            result = self.stub.Predict(request, 300.0)  # 10 secs timeout
        except Exception as e:
            print(e)
        lobprobs = (np.array(result.outputs['output0'].float_val))
        responses = []
        for j in range(1,4):
            responses.append([])
            labels = np.array(result.outputs['output'+str(j)].int64_val)
            for i in range(len(labels)):
                responses[-1].append(_get_string(labels[i,:]))
        return responses[0]
                    
                    
    def predict_list(self, image_list):
        result = {}
        for i, image in enumerate(image_list):
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'clreceipt'
            request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            request.inputs['images'].CopyFrom(
                tf.contrib.util.make_tensor_proto(image, shape=image.shape))
            request.inputs['width'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image.shape[1], shape=[1]))
            result_future = self.stub.Predict.future(request, 300.0)  # 10 secs timeout
            
            def _callback(result_future0, i=i):
                exception = result_future0.exception()
                if exception:
    #                 self.error_count += 1
                    print(exception)
                else:
                    sys.stdout.write(str(i))
                    sys.stdout.flush()
                    lobprobs = (numpy.array(result_future0.result().outputs['output0'].float_val))
                    responses = []
                    labels = ''
                    for j in range(1,4):
                        responses.append(numpy.array(
                            result_future0.result().outputs['output'+str(j)].int64_val))
                        labels += '@'+_get_string(responses[-1])
                    result[i] = labels
            print('push ' + str(i))
            result_future.add_done_callback(_callback)
        while len(result) < len(image_list):
            sleep(0.3)
            print('wait')
        return result
        
        
def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs['scores'].float_val)
            prediction = numpy.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


# def do_inference(hostport, work_dir, concurrency, num_tests):
#   """Tests PredictionService with concurrent requests.
#   Args:
#     hostport: Host:port address of the PredictionService.
#     work_dir: The full path of working directory for test data set.
#     concurrency: Maximum number of concurrent requests.
#     num_tests: Number of test images to use.
#   Returns:
#     The classification error rate.
#   Raises:
#     IOError: An error occurred processing test data set.
#   """
#   test_data_set = mnist_input_data.read_data_sets(work_dir).test
#   host, port = hostport.split(':')
#   channel = implementations.insecure_channel(host, int(port))
#   stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
#   result_counter = _ResultCounter(num_tests, concurrency)
#   for _ in range(num_tests):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'mnist'
#     request.model_spec.signature_name = 'predict_images'
#     image, label = test_data_set.next_batch(1)
#     request.inputs['images'].CopyFrom(
#         tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
#     result_counter.throttle()
#     result_future = stub.Predict.future(request, 5.0)  # 5 seconds
#     result_future.add_done_callback(
#         _create_rpc_callback(label[0], result_counter))
#   return result_counter.get_error_rate()
def _get_output(rnn_logits,sequence_length):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=False)

    return predictions

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([out_charset[c] for c in labels])
    return string

def simple_inference(hostport):
    tt = time()
    ra = ResultAccumulator(hostport)
    
    i = 2
    image = _get_image('/home/loitg/Downloads/real2/' + str(i) + '.JPG')
    batch = np.array([image]*1)
    rs = ra.predict_batch(batch)
    
#     image_list = []
#     for i in range(200):
#         try:
#             image = _get_image('/home/loitg/Downloads/real2/' + str(i) + '.JPG')
#         except  IOError as e:
#             print(e)
#             continue
#  
#         image_list.append(image)
#          
#     rs = ra.predict_list(image_list)
    
    
    
    print(rs)
    print(time() - tt)
        
     
    return 0
  
def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    #   error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
    #                             FLAGS.concurrency, FLAGS.num_tests)
    print(simple_inference(FLAGS.server))
#     print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
    tf.app.run()

'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
import sys
from time import sleep, time
from multiprocessing import Queue

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.batching.ops.gen_batch_ops import batch

from weinman import model, mjsynth
from weinman.validate import *


class TensorFlowPredictor(object):
    def __init__(self, hostport):
        self.host, self.port = hostport.split(':')
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
    
#         self._num_tests = num_tests
#         self._concurrency = concurrency
#         self._error = 0
#         self._done = 0
#         self._active = 0
#         self._condition = threading.Condition()
    
    def inc_error(self):
        with self._condition:
            self._error += 1
            
    def predict_batch(self, image_list):
        result = {}
        for i, image in enumerate(image_list):
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'clreceipt'
            request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            request.inputs['images'].CopyFrom(
                tf.contrib.util.make_tensor_proto(image, shape=image.shape))
            request.inputs['width'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image.shape[1], shape=[]))
            result_future = self.stub.Predict.future(request, 300.0)  # 10 secs timeout
            
            def _callback(result_future0, i=i):
                exception = result_future0.exception()
                if exception:
    #                 self.error_count += 1
                    print(exception)
                else:
                    sys.stdout.write(str(i))
                    sys.stdout.flush()
                    lobprobs = (np.array(result_future0.result().outputs['output0'].float_val))
                    responses = []
                    for j in range(1,4):
                        responses.append(np.array(
                            result_future0.result().outputs['output'+str(j)].int64_val))
                    labels = _get_string(responses[0])
                    result[i] = labels
            print('push ' + str(i))
            result_future.add_done_callback(_callback)
        while len(result) < len(image_list):
            sleep(0.3)
            print('wait')
        return result
    
    
class BatchLinePredictor(object):
    def __init__(self, server):
#         self.lock = threading.Lock()
        self.clientid, self.putq, self.getq = server.register()
        
    def predict(self, img_list):
        for i, img in enumerate(img_list):
            self.putq.put((str(i), time(), img))
        pred = {}
        waitcount = 0
        while True:
            topqueue = self.getq.get()
            if topqueue is not None:
                imgid, txt = topqueue
                pred[int(imgid)] = txt
                if len(pred) == len(img_list):
                    return pred
            else:
                sleep(0.5)
                waitcount += 1
                if waitcount > 100:
                    return pred
        
class Bucket(object):
    def __init__(self, maxtime, maxsize, widthrange):
        self.maxtime = maxtime
        self.maxsize = maxsize
        self.widthrange = widthrange
        self.imgs = []
        self.widths = []
        self.infos = []
        self.oldesttime = None

    
    def addImgToBucket(self, clientid, imgid, imgtime, img):
        if img.shape[1] > self.widthrange[0] and img.shape[1] <= self.widthrange[1]:
            newimg = cv2.copyMakeBorder(img, right=self.widthrange[1]-img.shape[1], value=0)
            print 'reshape before bucket ', newimg.shape
            self.imgs.append(newimg)
            self.widths.append(img.shape[1])
            self.infos.append((clientid, imgid))
            if self.oldesttime is None or imgtime < self.oldesttime:
                self.oldesttime = imgtime
            return True
        else:
            return False
    
    
    def getBatch(self):
        ready = False
        if len(self.imgs) > self.maxsize or (time() - self.oldesttime) > self.maxtime:
            batch = np.array(self.imgs)
            widths = np.array(self.widths)
            infos = self.infos
            self.imgs = []
            self.widths = []
            self.infos = []
            self.oldesttime = None
            return infos, batch, widths
        else:
            return None

class LocalServer(object):
    def __init__(self, modeldir):
        self.client_inputs = {}
        self.client_outputs = {}
        self.buckets = []
        #load graph
        self.graph = None
        self.maxclientid = 0
        self.modeldir = modeldir
    
    def register(self):
        self.maxclientid += 1
        clientid = str(self.maxclientid)
        self.client_inputs[clientid] = Queue()
        self.client_outputs[clientid] = Queue()
        return clientid, self.client_inputs[clientid], self.client_outputs[clientid]
        
    def run(self):
        with tf.Graph().as_default():
            with tf.device('/device:CPU:0'):
                image,width = _get_input() # Placeholder tensors
     
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
                restore_model(sess, _get_checkpoint(self.modeldir)) # Get latest checkpoint
                while True:
                    for id, clientq in self.client_inputs.iteritems():
                        imgid, imgtime, img = clientq.get()
                        self.addImgToBucket(id, imgid, imgtime, img)
                    
                    for bucket in self.buckets:
                        bckt = bucket.getBatch()
                        if bckt is not None:
                            infos, batch, widths = bckt
                            p = sess.run(predictions,{ image: batch, width: widths} )       
                            for (clientid, imgid), i in zip(infos, range(p[0].shape[0])):
                                txt = p[0][i,:]
                                txt = [i for i in txt if i >= 0]
                                txt = _get_string(txt)
                                self.client_outputs[clientid].put((imgid, txt))
                            
    
if __name__ == '__main__':
    pass
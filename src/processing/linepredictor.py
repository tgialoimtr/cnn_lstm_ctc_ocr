'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
import sys
import cv2
from time import sleep, time
from Queue import Empty, Full

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from weinman import model, mjsynth, validate


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
                    labels = validate._get_string(responses[0])
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
        
    def predict_batch(self, img_list):
        while True:
            try:
                self.putq.get(block=False)
            except Empty:
                break
        for i, img in enumerate(img_list):
#             print(str(time()) + ': putting ' + str(img.shape) + str(i) + ' to queue put ' + self.clientid)
            self.putq.put((str(i), time(), img), block=True)
        pred = {}
        waitcount = 0
        while True:
            try:
                topqueue = self.getq.get(timeout=0.5)
                imgid, txt = topqueue
                pred[int(imgid)] = txt
                if len(pred) == len(img_list):
                    return pred
            except Empty:
                waitcount += 1
#                 print(str(time()) + ': queue get ' + self.clientid + ' empty')
                if waitcount > 300:
                    for i in range(len(img_list)):
                        i = str(i)
                        if i not in pred:
                            pred[i] = ''
                            print 'PREDICTION TOO LONG, WILL RETURN EMPTY STRING !!!'
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
            newimg = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=self.widthrange[1]-img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
            if len(newimg.shape) < 3:
                newimg = newimg[:,:,np.newaxis]
            else:
                newimg = newimg[:,:,1]
            self.imgs.append(newimg)
            self.widths.append(img.shape[1])
            self.infos.append((clientid, imgid))
            if self.oldesttime is None or imgtime < self.oldesttime:
                self.oldesttime = imgtime
            return True
        else:
            return False
    
    
    def getBatch(self):
        if len(self.imgs) == 0: return None
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
    def __init__(self, modeldir, manager):
        self.client_inputs = {}
        self.client_outputs = {}
        self.buckets = []
        for w in range(32, 1000,32):
            self.buckets.append(Bucket(40,8,(w,w+32)))
        self.graph = None
        self.maxclientid = 0
        self.modeldir = modeldir
        self.manager = manager
    
    def register(self):
        print 'register called -------------------------------------------'
        self.maxclientid += 1
        clientid = str(self.maxclientid)
        self.client_inputs[clientid] = self.manager.Queue()
        self.client_outputs[clientid] = self.manager.Queue()
        return clientid, self.client_inputs[clientid], self.client_outputs[clientid]
        
    def run(self, states):
        with tf.Graph().as_default():
            with tf.device('/device:CPU:0'):
                image,width = validate._get_input() # Placeholder tensors
     
                proc_image = validate._preprocess_image(image)
    
            with tf.device('/device:CPU:0'):
                features,sequence_length = model.convnet_layers( proc_image, width, 
                                                                 validate.mode)
                logits = model.rnn_layers( features, sequence_length,
                                           mjsynth.num_classes() )
            with tf.device('/device:CPU:0'):
                predictions = validate._get_output( logits,sequence_length)
    
                session_config = validate._get_session_config()
                restore_model = validate._get_init_trained()
            
                init_op = tf.group( tf.global_variables_initializer(),
                                tf.local_variables_initializer()) 
            with tf.Session(config=session_config) as sess:
            
                sess.run(init_op)
                restore_model(sess, validate._get_checkpoint(self.modeldir)) # Get latest checkpoint
                print(str(time()) + 'server started, waiting image ...') 
                states['server_started'] = True
                while True:
                    for clientid, clientq in self.client_inputs.iteritems():
                        try:
                            imgid, imgtime, img = clientq.get(block=False)
                            success_count = 0
                            for bucket in self.buckets:
                                success = bucket.addImgToBucket(clientid, imgid, imgtime, img)
                                if success: 
                                    success_count += 1
#                                     print(str(time()) + ': image ' + str(img.shape) + ' from ' + clientid +' add successful to bucket ' + str(bucket.widthrange))
                            assert(success_count==1)
                        except Empty:
#                             print(str(time()) + ': queue put ' + clientid + ' empty')
                            sleep(0.1)

                    
                    for bucket in self.buckets:
                        bckt = bucket.getBatch()
                        if bckt is not None:
                            infos, batch, widths = bckt
                            tt = time()
                            print(str(time()) + ': BATCH INFO --------------')
                            print infos
                            print batch.shape, np.mean(widths)
                            p = sess.run(predictions,{ image: batch, width: widths} )
                            print '-------------------' + str(time() - tt)    
                            for (clientid, imgid), i in zip(infos, range(p[0].shape[0])):
                                txt = p[0][i,:]
                                txt = [i for i in txt if i >= 0]
                                txt = validate._get_string(txt)
                                try:
                                    self.client_outputs[clientid].put((imgid, txt), block=False)
                                except Full:
                                    print(str(time()) + ': queue get ' + clientid + ' full')
                            
    
if __name__ == '__main__':
    pass
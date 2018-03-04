'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
import cv2
from time import sleep, time
from Queue import Empty, Full

import numpy as np
import tensorflow as tf

from weinman import model, mjsynth, validate
from common import args
        
class Bucket(object):
    def __init__(self, maxtime, batchsize, widthrange):
        self.maxtime = maxtime
        self.batchsize = batchsize
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
        if len(self.imgs) > self.batchsize or (time() - self.oldesttime) > self.maxtime:
            batch = np.array(self.imgs[:self.batchsize])
            widths = np.array(self.widths[:self.batchsize])
            infos = self.infos[:self.batchsize]
            self.imgs = self.imgs[self.batchsize:]
            self.widths = self.widths[self.batchsize:]
            self.infos = self.infos[self.batchsize:]
            self.oldesttime = time() #fix this maybe
            return infos, batch, widths
        else:
            return None

class LocalServer(object):
    def __init__(self, modeldir, manager):
        self.client_inputs = {}
        self.client_outputs = {}
        self.buckets = []
        for w in range(32, 1000,32):
            self.buckets.append(Bucket(args.bucket_max_time, args.bucket_size,(w,w+32)))
        self.graph = None
        self.maxclientid = 0
        self.modeldir = modeldir
        self.manager = manager
    
    def register(self):
        self.maxclientid += 1
        clientid = str(self.maxclientid)
        self.client_inputs[clientid] = self.manager.Queue()
        self.client_outputs[clientid] = self.manager.Queue()
        return clientid, self.client_inputs[clientid], self.client_outputs[clientid]
        
    def run(self, states, logger):
        with tf.Graph().as_default():
            image,width = validate._get_input(args.bucket_size) # Placeholder tensors
 
            proc_image = validate._preprocess_image(image)

            with tf.device(args.device):
                features,sequence_length = model.convnet_layers( proc_image, width, 
                                                                 validate.mode)
                logits = model.rnn_layers( features, sequence_length,
                                           mjsynth.num_classes() )
                predictions = validate._get_output( logits,sequence_length)
    
            session_config = validate._get_session_config()
            restore_model = validate._get_init_trained()
        
            init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 
            with tf.Session(config=session_config) as sess:
            
                sess.run(init_op)
                restore_model(sess, validate._get_checkpoint(self.modeldir)) # Get latest checkpoint
                logger.info('%s, server started, waiting image ...', str(time()))
#                 print(str(time()) + 'server started, waiting image ...') 
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
                            width_mean = np.mean(widths)
                            if batch.shape[0] < args.bucket_size:
                                additional_batch = np.zeros(shape=(args.bucket_size - batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]), dtype=np.float32)
                                additional_width = np.ones(shape=(args.bucket_size - batch.shape[0],), dtype=np.int32)*widths[0]
                                infos += [('0','0')]*(args.bucket_size - batch.shape[0])
                                batch = np.concatenate((batch, additional_batch))
                                widths = np.concatenate((widths, additional_width))
                            tt = time()
                            p = sess.run(predictions,{ image: batch, width: widths} )
                            logger.debug('batch info: %d, %s, %6.2f; timeL%6.6f', args.bucket_size - batch.shape[0], str(batch.shape[2]), width_mean, (time() - tt)/batch.shape[2])
                            for (clientid, imgid), i in zip(infos, range(p[0].shape[0])):
                                if clientid == '0': continue
                                txt = p[0][i,:]
                                txt = [i for i in txt if i >= 0]
                                txt = validate._get_string(txt)
                                try:
                                    self.client_outputs[clientid].put((imgid, txt), block=False)
                                except Full:
                                    log.warning('queue get %d full', clientid)
                    
                    sleep(0.5)
    
if __name__ == '__main__':
    pass
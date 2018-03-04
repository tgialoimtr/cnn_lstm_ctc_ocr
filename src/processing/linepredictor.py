'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
from time import time
from Queue import Empty, Full
from common import args
    
class BatchLinePredictor(object):
    def __init__(self, server, logger):
#         self.lock = threading.Lock()
        self.clientid, self.putq, self.getq = server.register()
        logger.info('receive clientid %s', self.clientid)
        
    def predict_batch(self, img_list, logger):
        while True:
            try:
                self.putq.get(block=False)
            except Empty:
                break
        for i, img in enumerate(img_list):
            self.putq.put((str(i), time(), img), block=True)
        logger.debug('put %d imgs to queue put %s', len(img_list), self.clientid)
        pred = {}
        waitcount = 0
        a = 1.0/0.0
        while True:
            try:
                topqueue = self.getq.get(timeout=args.qget_wait_interval)
                imgid, txt = topqueue
                pred[int(imgid)] = txt
                if len(pred) == len(img_list):
                    return pred
            except Empty:
                waitcount += 1
#                 print(str(time()) + ': queue get ' + self.clientid + ' empty')
                if waitcount > args.qget_wait_count:
                    for i in range(len(img_list)):
                        if i not in pred:
                            pred[i] = ''
                            print 'PREDICTION TOO LONG, WILL RETURN EMPTY STRING !!!'
                    return pred
                            
    
if __name__ == '__main__':
    pass
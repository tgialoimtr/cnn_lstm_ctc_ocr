'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
from time import time
from Queue import Empty, Full

class DFO(object):
    pass

args = DFO()
args.qget_wait_count = 200
args.qget_wait_interval = 0.5
    
    
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
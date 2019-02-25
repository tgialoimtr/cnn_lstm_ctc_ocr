#!/usr/bin/env python
import sys
sys.path += ['/usr/local/lib/python2.7/dist-packages/mininet-2.2.1-py2.7.egg', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/loitg/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0']
import cv2
from pylab import *
from scipy.ndimage import morphology
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, uniform_filter1d
import ocrolib
from skimage.filters import threshold_sauvola

from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import time
import datetime

from linepredictor import BatchLinePredictor
from lineextractor import extractLines2
from sklearn.externals import joblib
from common import args

   
def sauvola(grayimg, w=51, k=0.2, scaledown=None, reverse=False):
    mask =None
    if scaledown is not None:
        mask = cv2.resize(grayimg,None,fx=scaledown,fy=scaledown)
        w = int(w * scaledown)
        if w % 2 == 0: w += 1
        if w == 1: w=3
        mask = threshold_sauvola(mask, w, k)
        mask = cv2.resize(mask,(grayimg.shape[1],grayimg.shape[0]),fx=scaledown,fy=scaledown)
    else:
        if w % 2 == 0: w += 1
        if w == 1: w=3
        mask = threshold_sauvola(grayimg, w, k)
    if reverse:
        return where(grayimg > mask, uint8(0), uint8(1))
    else:
        return where(grayimg > mask, uint8(1), uint8(0)) 


def calc_line(oriline):
    line = sauvola(oriline,w=oriline.shape[0]/2, k=0.05, reverse=True)
    oridense = '{:3.3f}'.format(mean(oriline))
    dens = '{:3.3f}'.format(mean(line))
#     rati = '{:3.3f}'.format(1.0*line.shape[1]/line.shape[0])
    _,n = morph.label(line)
    n = '{:3.3f}'.format(1.0*n/oriline.shape[1]*oriline.shape[0])
    return dens+'_'+n+'_'+oridense, line

def pre_check_line(oriline):
    if oriline.shape[0] < 10:
        return False
    if 1.0*oriline.shape[1]/oriline.shape[0] < 1.28:
        return False
    if 1.0*oriline.shape[1]/oriline.shape[0] > 23.4:
        return False
#     if mean(oriline) < 0.35:
#         return False
#     if oriline.shape[0] > 25:
#         line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True, scaledown=20.0/oriline.shape[0])
#     else:
#         line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True)
#     if mean(line) < 0.15:
#         return False
#     _,n = morph.label(line)
#     n = 1.0*n/oriline.shape[1]*oriline.shape[0]
#     if n > 15:
#         return False
    return True


class PagePredictor:
    def __init__(self, localserver, logger):
#         self.lock = threading.Lock()
        if localserver is not None:
            self.linepredictor = BatchLinePredictor(localserver, logger)
        self.clf = joblib.load(args.model_path + 'le_model_3.pkl')
        
    def ocrImage(self, imgpath, logger):
        lines, dic4bounds, illu, img2 = extractLines2(imgpath, self.clf)

        # directory='/tmp/lines/'+imgpath.split('/')[-1]
        # print(directory)
        # try:
            # os.stat(directory)
        # except:
            # os.mkdir(directory) 
        # cv2.imwrite(directory + '/' + 'debugbox.JPG', illu)
        # cv2.imwrite(directory + '/' + 'debugline.JPG', img2)

                
        location_text = []
        line_list = []
        posy_list = []
        for i,line in enumerate(lines):
#             hihi, sau = calc_line(line)

            if not pre_check_line(line): continue
#             cv2.rectangle(illubox, (l.bounds[1].start, l.bounds[0].start), 
#                           (l.bounds[1].stop, l.bounds[0].stop), 1, 2)
            newwidth = int(32.0/line.shape[0] * line.shape[1])
            if newwidth < 32 or newwidth > 1000: continue
            line = cv2.resize(line, (newwidth, 32))
            line = (line*255).astype(np.uint8)
            
            # cv2.imwrite(directory + '/' + str(i) + '.JPG', line)
            
            line_list.append(line)
            _,y1 = dic4bounds[i]['botright']
            posy_list.append(y1) #whatever nearer to middle line

        if len(line_list) == 0: raise ValueError('Image contains no contents.')
        batchname = datetime.datetime.now().isoformat()
        pred_dict = self.linepredictor.predict_batch(batchname, line_list, logger)
        logger.debug('%s', str(pred_dict))
        
        
        for i in range(len(line_list)):
            location_text.append((posy_list[i], pred_dict[i]))

        location_text.sort()   
        lines = []
        for i, result in enumerate(location_text): 
            lines.append(result[1])
        return lines, 0
                    
#                     
# if __name__ == "__main__":
#     import os
# 
#         manager = Manager()
#         states = manager.dict()
#     server = LocalServer(args.model_path, manager)
# 
# def runserver(server, states):
#     logger = createLogger('server')
#     server.run(states, logger)
#     
#     pp = PagePredictor(None, None)
#     with open('/home/loitg/Downloads/complex-bg/java3/rs.txt', 'w') as rs:
#         for filename in os.listdir('/home/loitg/Downloads/complex-bg/java/'):        
#             if filename[-3:].upper() == 'JPG':
#                 
#                 tt = time.time()
#                 ret = pp.ocrImage('/home/loitg/Downloads/complex-bg/java/' + filename, None)
#                 rs.write(filename + '----------------' + str(time.time() - tt) + '\n')
#                 rs.write(ret+ '\n')
#                 rs.flush()

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
    if mean(oriline) < 0.35:
        return False
    if oriline.shape[0] > 25:
        line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True, scaledown=20.0/oriline.shape[0])
    else:
        line = sauvola(oriline,w=oriline.shape[0]*3/4, k=0.05, reverse=True)
    if mean(line) < 0.15:
        return False
    _,n = morph.label(line)
    n = 1.0*n/oriline.shape[1]*oriline.shape[0]
    if n > 15:
        return False
    return True


class PagePredictor:
    def __init__(self, localserver, logger):
#         self.lock = threading.Lock()
        if localserver is not None:
            self.linepredictor = BatchLinePredictor(localserver, logger)
    
    def ocrImage(self, imgpath, logger):
        tt=time.time()
        
        img_grey = ocrolib.read_image_gray(imgpath)
        (h, w) = img_grey.shape[:2]
        img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
#             cv2.imshow('debug', img00)
#             cv2.waitKey(-1)
        angle = estimate_skew_angle(img00,linspace(-5,5,42))
        print 'goc', angle
    
        rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
        
        h,w = img_grey.shape
        img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.999, cv2.NORM_MINMAX)
        binary = sauvola(img_grey, w=50, k=0.05, scaledown=0.2, reverse=True)
    
        binaryary = morph.r_closing(binary[h/4:3*h/4,w/4:3*w/4].astype(bool), (args.connect,1))
        labels,n = morph.label(binaryary)
        objects = morph.find_objects(labels) ### <<<==== objects here
        bysize = sorted(objects,key=sl.area)
        scalemap = zeros(binaryary.shape)
        for o in bysize:
            if amax(scalemap[o])>0: continue
            scalemap[o] = sl.area(o)**0.5
        scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here

    
        location_text = []
        line_list = []
        bounds_list = []
        illubox = np.zeros_like(img_grey, dtype=np.uint8)
        for i,l in enumerate(lines):
            line = extract_line(img_grey,l,pad=args.pad)
#             hihi, sau = calc_line(line)
            if not pre_check_line(line): continue
            cv2.rectangle(illubox, (l.bounds[1].start, l.bounds[0].start), 
                          (l.bounds[1].stop, l.bounds[0].stop), 1, 2)
            newwidth = int(32.0/line.shape[0] * line.shape[1])
            if newwidth < 32 or newwidth > 1000: continue
            line = cv2.resize(line, (newwidth, 32))
            line = (line*255).astype(np.uint8)
            line_list.append(line)
            bounds_list.append(l.bounds)
            
#             directory='/tmp/temp_hope/'+imgpath.split('/')[-1]
#             print(directory)
#             try:
#                 os.stat(directory)
#             except:
#                 os.mkdir(directory) 
#             cv2.imwrite(directory+'/'+ str(i) + '_' + hihi +'.JPG', line)
#             cv2.imwrite(directory+'/'+ str(i) + '_' + hihi +'_sau.JPG', sau*255)
#                
#         return 'hihi'
        image = transpose(array([seeds0, illubox, img_grey]),[1,2,0])
        directory='/home/loitg/Downloads/complex-bg/java3/'+imgpath.split('/')[-1]
        cv2.imwrite(directory + '_3.jpg', (image*255).astype(np.uint8))
        return ''
        qualityCode = imgquality(line_list, bounds_list, logger)
        if len(line_list) == 0: raise ValueError('Image contains no contents.')
        batchname = datetime.datetime.now().isoformat()
        pred_dict = self.linepredictor.predict_batch(batchname, line_list, logger)
        logger.debug('%s', str(pred_dict))
        for i in range(len(line_list)):
            result = psegutils.record(bounds = bounds_list[i], text=pred_dict[i], available=True)
            location_text.append(result)

        location_text.sort(key=lambda x: x.bounds[1].stop)
        i = 0
        while i < len(location_text):
            result = location_text[i]
            if result.available:
                linemap = []
                
                for j in range(i, len(location_text)):
                    if j==i: continue
                    candidate = location_text[j]
                    if not candidate.available: continue
                    current_height = result.bounds[0].stop - result.bounds[0].start
                    sameline = abs(result.bounds[0].stop - candidate.bounds[0].stop)
                    rightness = candidate.bounds[1].start - result.bounds[1].stop
                    if sameline < 0.5*current_height and rightness > -current_height:
                        linemap.append((sameline**2 + rightness**2, candidate))
                if len(linemap) > 0:
                    j, candidate = min(linemap)
                    result.text += (' ' + candidate.text)
                    yy = slice(minimum(candidate.bounds[0].start, result.bounds[0].start), maximum(candidate.bounds[0].stop, result.bounds[0].stop))
                    xx = slice(minimum(candidate.bounds[1].start, result.bounds[1].start), maximum(candidate.bounds[1].stop, result.bounds[1].stop))
                    result.bounds = (yy,xx)
                    candidate.available = False
                    continue
                else:
                    i+=1
                    continue
            else:
                i+=1
                continue       
            
        location_text.sort(key=lambda x: x.bounds[0].stop)   
        lines = []
        for i, result in enumerate(location_text): 
            if result.available:
                lines.append(result.text)
    #             ocrolib.write_text(args.outtext+str(i)+".txt",pred)/home/loitg/Downloads/complex-bg
        return lines, qualityCode
                    
                    
if __name__ == "__main__":
    import os

        manager = Manager()
        states = manager.dict()
    server = LocalServer(args.model_path, manager)

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger)
    
    pp = PagePredictor(None, None)
    with open('/home/loitg/Downloads/complex-bg/java3/rs.txt', 'w') as rs:
        for filename in os.listdir('/home/loitg/Downloads/complex-bg/java/'):        
            if filename[-3:].upper() == 'JPG':
                
                tt = time.time()
                ret = pp.ocrImage('/home/loitg/Downloads/complex-bg/java/' + filename, None)
                rs.write(filename + '----------------' + str(time.time() - tt) + '\n')
                rs.write(ret+ '\n')
                rs.flush()

       
#     tt = time.time() 
#     ret = PagePredictor(sys.argv[1]).ocrImage(sys.argv[2])
#     with open(sys.argv[3], 'w') as outputfile:
#         outputfile.write(ret)      
#     print(time.time() -tt)

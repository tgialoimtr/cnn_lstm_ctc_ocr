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
from imgquality import imgquality
from test_charboxfind import findBox
import random
       
class obj:
    def __init__(self):
        pass
    

args = obj()
args.zoom = 0.5
args.range = 20
args.debug = 1
args.perc= 80
args.escale = 1.0
args.threshold = 0.5
args.lo = 5
args.hi = 90
args.usegauss = False
args.vscale = 1.0
args.hscale = 1.0
args.threshold = 0.25
args.pad = 1
args.expand = 3
args.model = '/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz'
args.inputdir = '/root/ocrapp/tmp/cleanResult/'
args.connect = 1
args.noise = 8

def rotate(image, angle):
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
    dst = cv2.warpAffine(image, M, image.shape)
    return dst
def resizeToHeight(img, newheight):
    newwidth = int(1.0*img.shape[1]/img.shape[0]*newheight)
    newimg = cv2.resize(img, (newwidth, newheight))
    return newimg
def summarize(a):
    b = a.ravel()
    return len(b),[amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])
   
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

def estimate_skew_angle(image,angles):
    estimates = []
    binimage = sauvola(image, 11, 0.1).astype(float)
    for a in angles:
        rotM = cv2.getRotationMatrix2D((binimage.shape[1]/2,binimage.shape[0]/2),a,1)
        rotated = cv2.warpAffine(binimage,rotM,(binimage.shape[1],binimage.shape[0]))
        v = mean(rotated,axis=1)
        d = [abs(v[i] - v[i-1]) for i in range(1,len(v))]
        d = var(d)
        estimates.append((d,a))
#     if args.debug>0:
#         plot([y for x,y in estimates],[x for x,y in estimates])
#         ginput(1,args.debug)
    _,a = max(estimates)
    return a


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

def linfit(xs, ys):
    n = len(xs)
    sumx = 0; sumy = 0; sumxy = 0; sumx2 = 0; sumy2 = 0; 
    for i in range(n):
        xi = xs[i]; yi = ys[i]
        sumx += xi
        sumy += yi
        sumx2 += xi*xi
        sumy2 += yi*yi
        sumxy += xi*yi
    denom = (n*sumx2 - sumx*sumx)
    
    b = (n*sumxy - sumx*sumy) / denom
    m = (sumy*sumx2 - sumx*sumxy) /denom
    s2e = (n*sumy2 - sumy*sumy - b*b*denom)
    
    return b,m,s2e

# class DistVar(object):
#     def __init__(self):
#         self.val
#         self.std

class Node(object):
    def __init__(self, *obj):
        if len(obj) == 2:
            pass
        else:
            pass
        self.top
        self.bottom
        self.height
    
class SubLine(object):
    LOOKAHEAD = 2.5
    
    class Config(object):
        def __init__(self):
            pass
        
        def add
        
    ###  <<<<<<<<<<<<<<<<------------------
    def __init__(self, initbound):
        height = initbound[0].stop - initbound[0].start
        x = (initbound[1].start + initbound[1].stop)/2
        top = (x, initbound[0].start)
        bottom = (x, initbound[0].stop)
        self.nodes = [Node(top, bottom, height)]
#         self.height = DistVar(height, height/2)
#         self.angel = DistVar(0,45)
#         self.baseline
        self.curpos = bottom
        self.isnew = True
    ###  ------------------>>>>>>>>>>>>>>>>>>
    
    def score(self, conf):
        pass

    def nextRange(self):
        x1 = self.curpos[0]
        x2 = x1 + self.height * self.LOOKAHEAD
        for x in range(x1,x2):
            y = self.curpos[1]
            y1 = y - (x-x1); y2 = y + (x-x1)
            for y in range(y1,y2):
                yield x, y
    
    ###  <<<<<<<<<<<<<<<<------------------        
    def suggest(self, allnodes, allpoints):
        for x,y in self.nextRange():
            if allnodes[x][y] == None: continue
            calc angle
            add to angle bin
            
        return [conf1, conf2]
    ###  ------------------>>>>>>>>>>>>>>>>>>
    
    
    def next(self, allnodes, allpoints):
        confs = self.suggest(allnodes, allpoints)
        results = []
        for conf in confs:
            score = self.score(conf)

            results.append((score, conf))
        
        ### iterate results to select:
        
        conf to another subline
        
        
    def draw(self, img):
        pass
    
class Abc(object):
    def __init__(self, img_grey, illu_scale):
        self.img_grey = img_grey
        self.illu = cv2.cvtColor(img_grey.astype(np.float32), cv2.COLOR_GRAY2BGR)
        self.illu = cv2.resize(self.illu, None, fx=self.illu_scale, fy=self.illu_scale)
        self.illu = (self.illu*255).astype(np.uint8)             

    def drawLine(self, subline):
        subline.param *= 2
        subline.draw(self.illu)
    
    def getIllu(self):
        return self.illu
    
def extractLines2(imgpath):
    img_grey = ocrolib.read_image_gray(imgpath)
    img_grey = img_grey[:img_grey.shape[0]/2,:]
    
    (h, w) = img_grey.shape[:2]
    img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
    angle = estimate_skew_angle(img00,linspace(-5,5,42))
    print 'goc', angle
    rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
    
    h,w = img_grey.shape
    img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.999, cv2.NORM_MINMAX)
    
    illustrator = Abc(img_grey, 2.0)
    objects, scale = findBox(img_grey)
    
    nodes = [[None for j in range(w)] for i in range(h)]
    points = [[0 for j in range(w)] for i in range(h)]
    for bound in objects:
        top = ((bound[1].start + bound[1].stop)/2, bound[0].start)
        bottom = ((bound[1].start + bound[1].stop)/2, bound[0].stop)
        points[bottom[0]][bottom[1]] = 1
        points[top[0]][top[1]] = 1
        nodes[bottom[0]][bottom[1]] = Node(top, bottom, bound[0].stop - bound[0].start)
    
    allines = []
    
    def move(subline, allnodes, allpoints):
        newsublines = subline.next(allnodes, allpoints)
        if len(newsublines) > 0:
            for new in newsublines:
                if new.isnew: 
                    allines.appends(new)
                    new.isnew = False
                move(new, allnodes, allpoints)

    for bound in objects:
        subline = SubLine(bound)
        allines.appends(subline)
        subline.isnew = False
        move(subline, nodes, points)
    
    for line in allines:
        illustrator.draw(line)
    
    
    return img_grey, illustrator.getIllu()

    
import os
if __name__ == "__main__":
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/tmp/'):        
        if filename[-3:].upper() == 'JPG':
            print filename
            img, illu = extractLines2('/home/loitg/Downloads/complex-bg/tmp/' + filename)
            cv2.imshow('illu', illu)
            cv2.waitKey(-1)


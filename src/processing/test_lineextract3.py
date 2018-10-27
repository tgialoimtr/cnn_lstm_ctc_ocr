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
import math
from statsmodels.nonparametric.smoothers_lowess import lowess
       
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
    ISTOP = 2
    ISBOT = 1
    ISEMPTY = 0
    IGNORED = -1
    INTERVALS = [0.2,0.3,0.5]
    LINE_IOU_COMBINE_THRESHOLD = 0.8
    LINE_SCORE_THRESHOLD = 0.5
    
    class Config1(object):
        
        MIN_INSIDER = 3
        MAX_ABOVE = 2
        MAX_OUTSIDER_RATIO = 0.2
        
        def __init__(self, topy, boty, x):
            # pre-set
            self.atopy
            self.aboty
            self.ax
            # post-set
            self.angle
            self.toppoints = []
            self.botpoints = []
            # others
            height = float(self.aboty - self.atopy)
            self.heightchkpts = []
            self.heightchkpts.append(-height*(self.INTERVALS[0] + self.INTERVALS[1]))#int0
            self.heightchkpts.append(-height*(self.INTERVALS[1]))#int1
            self.heightchkpts.append(0.0)#int2
            self.heightchkpts.append(self.INTERVALS[2] * height)#int3
            self.heightchkpts.append(height - self.INTERVALS[2] * height)#int4
            self.heightchkpts.append(height)#int5
            self.heightchkpts.append(height + height*self.INTERVALS[1])#int6
            self.heightchkpts.append(height + height*(self.INTERVALS[1] + self.INTERVALS[0]))#int7
            self.intervals = [(self.heightchkpts[i], self.heightchkpts[i+1]) for i in range(0,7)] # 012--456
            self.topsinintervals = [[]] * len(self.intervals)
            self.botsinintervals = [[]] * len(self.intervals)
            self.sin = {}; self.cos = {}
            for a in self.angle_list:
                self.sin[a] = math.sin(a/180.0*math.pi)
                self.cos[a] = math.cos(a/180.0*math.pi)
        
        def _add2intervals(self, y1, point, point_type):
            for i, interval in enumerate(self.intervals):
                if y1 >= interval[0] and y1 < interval[1]:
                    if point_type == self.ISTOP:
                        self.topsinintervals[i].append(point)
                    if point_type == self.ISBOT:
                        self.botsinintervals[i].append(point)
        
        def _intervalHasLine(self):
            top_in_0 = len(self.topsinintervals[0])
            top_in_12 = len(self.topsinintervals[1]) + len(self.topsinintervals[2])
            bot_in_12 = len(self.botsinintervals[1]) + len(self.botsinintervals[2])
            top_in_45 = len(self.topsinintervals[4]) + len(self.topsinintervals[5])
            bot_in_45 = len(self.botsinintervals[4]) + len(self.botsinintervals[5])
            bot_in_6 = len(self.botsinintervals[6])
            if (top_in_12 > self.MIN_INSIDER and top_in_0 < self.MAX_ABOVE and 1.0*bot_in_12/top_in_12 < self.MAX_OUTSIDER_RATIO) and \
                (bot_in_6 > self.MIN_INSIDER and bot_in_45 < self.MAX_ABOVE and 1.0*top_in_45/bot_in_45 < self.MAX_OUTSIDER_RATIO):
                return True
            else:
                return False
            
        def add(self, x,y, point_type):
            if point_type == self.ISTOP:
                self.toppoints.append((x,y))
            elif point_type == self.ISBOT:
                self.botpoints.append((x,y))
            
        def finalize(self):
            MAX_ANGLE = 30.0
            rets = []
            for a in np.linspace(-MAX_ANGLE, MAX_ANGLE, 5):
                self.topsinintervals = [[]] * len(self.intervals)
                self.botsinintervals = [[]] * len(self.intervals)
                for point in self.toppoints:
                    x1 = point[0] - self.ax
                    y1 = point[1] - self.atopy                    
                    ytop_proj = y1 + x1 * self.sin[a] / self.cos[a]
                    self._add2intervals(ytop_proj, SubLine.ISTOP)
                for point in self.botpoints:
                    x1 = point[0] - self.ax
                    y1 = point[1] - self.atopy                    
                    ybot_proj = y1 + x1 * self.sin[a] / self.cos[a]
                    self._add2intervals(ybot_proj, SubLine.ISBOT)
                if self._intervalHasLine():
                    conf = SubLine.Config1(self.atopy, self.aboty, self.ax)
                    conf.angle = self.angle
                    conf.toppoints = self.topsinintervals[1] + self.topsinintervals[2]
                    conf.botpoints = self.botsinintervals[4] + self.botsinintervals[5]
                    rets.append(conf)
                
            return rets
                
    ###  <<<<<<<<<<<<<<<<------------------
    def __init__(self, topy, boty, x, tops=[], bottoms=[]):      
        self.tops = tops
        self.bottoms = bottoms
        self.curtopy = topy
        self.curboty = boty
        self.height = boty-topy
        self.curx = x
        self.isnew = True
        self.nextCount = 0
        self.lastytopindex = 0
        self.lastybotindex = 0

    def _initConfig(self):
        possible_confs = []
        possx = self.curx
        posstop = self.curtopy
        possbot = self.curboty
        possible_confs.append(SubLine.Config1(posstop, possbot, possx))
#         d = self.height/3
#         possible_confs.append(SubLine.Config1(posstop+d, possbot+d, possx))
#         possible_confs.append(SubLine.Config1(posstop-d, possbot-d, possx))
#         possible_confs.append(SubLine.Config1(posstop+d, possbot+d, possx + d))
#         possible_confs.append(SubLine.Config1(posstop, possbot, possx + d))
#         possible_confs.append(SubLine.Config1(posstop-d, possbot-d, possx + d))
        return possible_confs
    
    def suggest1(self, allnodes, allpoints):
        # Generate config
        allconfs = self._initConfig()
        retconfs = []
        for x,y in self.nextRange():
            point_type = allpoints[x][y]
            if point_type == self.ISEMPTY: continue
            for cand_conf in allconfs:
                cand_conf.add(x,y,point_type)
        for cand_conf in allconfs:
            confs = cand_conf.finalize()
            if confs is not None and len(confs) > 0:
                retconfs += confs
        return retconfs
    ###  ------------------>>>>>>>>>>>>>>>>>>
    
    def _score(self, lefts, rights, fromindexleft=0):
        lefts = np.array(lefts)
        rights = np.array(rights)
        
        xs = np.concatenate((lefts[fromindexleft:,0], rights[:,0]))
        ys = np.concatenate((lefts[fromindexleft:,1], rights[:,1]))
        
        yhat = lowess(ys, xs, frac=0.5)
        error = np.sum(np.square(yhat - ys))
        leftmostpoint = (xs[0], yhat[0])
        rightmostpoint = (xs[-1], yhat[-1])
        
        return error, leftmostpoint, rightmostpoint
        
    def score(self, conf):
        toperror, topleft, topright = self._score(self.tops, conf.toppoints, self.lastytopindex)
        boterror, botleft, botright = self._score(self.bots, conf.bottompoints, self.lastybotindex)
        return toperror + boterror, [topleft, topright, botright, botleft]

    def nextRange(self):
        x1 = self.curx
        x2 = x1 + self.height * self.LOOKAHEAD
        for x in range(x1,x2):
            y = self.curboty
            y1 = y - (x-x1); y2 = y + (x-x1)
            for y in range(y1,y2):
                yield x, y
    

    def iou(self, poly1, poly2):
        poly1 = Polygon(poly1)
        poly2 = Polygon(poly2)
        i = poly1.intersection(poly2).area()
        o = poly1.union(poly2).area()
        return 1.0*i/o
    
    def _filterCombineConf(self, results):
        n = len(results)
        for i in range(0, n):
            for j in range(i, n):
                iou = self.iou(results[i][2], results[j][2])
                if iou > SubLine.LINE_IOU_COMBINE_THRESHOLD:
                    if results[j][0] > results[i][0]:
                        results[i] = results[j]
                    results[j] = None
        return results

    def next(self, allnodes, allpoints):
        confs = self.suggest1(allnodes, allpoints)
        results = []
        for conf in confs:
            score, confshape = self.score(conf)

            results.append((score, conf, confshape))
        
        results = self._filterCombineConf(results)
        selected_confs_shapes = [(x[1],x[2]) for x in results if x is not None and x[0] > SubLine.LINE_SCORE_THRESHOLD]
        retlines = []
        for i, (conf, confshape) in enumerate(selected_confs_shapes):
            topright = confshape[1]
            botright = confshape[2]
            topy = topright[1]; boty = botright[1]; 
            x = (topright[0] + botright[0])/2
            if i == 0:
                self.curtopy = topy
                self.curboty = boty
                self.curx = x
                self.lastytopindex = len(self.tops)
                self.lastybotindex = len(self.bots)
                self.tops += conf.toppoints
                self.bots += conf.botpoints
                retlines.append(self)
            else:
                another = SubLine(topy=topy,
                                  boty=boty,
                                  x=x,
                                  tops=self.tops + conf.toppoints,
                                  bottoms=self.bots + conf.botpoints)
                another.lastytopindex = len(self.tops)
                another.lastybotindex = len(self.bots)  
                another.isnew = True
                another.nextCount = self.nextCount + 1
                retlines.append(another)
        
            for point in conf.toppoints: 
                allpoints[point[0]][point[1]] = SubLine.IGNORED
        
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
    points = [[SubLine.ISEMPTY for j in range(w)] for i in range(h)]
    for bound in objects:
        top = ((bound[1].start + bound[1].stop)/2, bound[0].start)
        bottom = ((bound[1].start + bound[1].stop)/2, bound[0].stop)
        points[bottom[0]][bottom[1]] = SubLine.ISBOT
        points[top[0]][top[1]] = SubLine.ISTOP
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

    for bound in objects: # sorted
        subline = SubLine(topy=bound[0].start,
                          boty=bound[0].stop,
                          x=(bound[1].start + bound[1].stop)/2)
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

